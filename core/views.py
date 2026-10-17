import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from functools import wraps

import cv2
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.db.models import Count
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import autoseg, exports, geometry, importers, jobs, ml, spectral
from .models import Annotation, Project, ROIMeasurement, Sample

YOLO_KIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_kit")
YOLO_KIT_FILES = ["LEIA-ME.txt", "requirements.txt", "yolo_trainer_gui.py", "iniciar_treino.bat",
                  "iniciar_treino.sh", "logo.png", "icon.png"]


# ── Acesso ────────────────────────────────────────────────────────────────────
# Cada usuario so ve os proprios projetos; superusuario ve todos (mesmo
# privilegio do Django admin). Tudo que recebe um id do cliente passa por aqui.

def _projects(request):
    qs = Project.objects.select_related("owner")
    return qs if request.user.is_superuser else qs.filter(owner=request.user)


def _project(request, pid):
    return get_object_or_404(_projects(request), id=pid)


def _sample(request, sid):
    return get_object_or_404(Sample.objects.select_related("project"), id=sid, project__in=_projects(request))


def _body(request):
    try:
        return json.loads(request.body or "{}")
    except ValueError:
        return {}


def _err(msg, status=400):
    return JsonResponse({"ok": False, "error": msg}, status=status)


def json_errors(view):
    """Erros esperados (formato, falta de anotacao...) viram JSON com mensagem."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except (autoseg.AutoSegError, ml.MLError, exports.ExportError, importers.ImportError_) as e:
            return _err(str(e))
        except FileNotFoundError as e:
            return _err(f"Arquivo da amostra não encontrado ({e}). Se ela foi importada vinculada, "
                        "a pasta original pode ter sido movida.", 409)
    return wrapper


# ── Projetos ──────────────────────────────────────────────────────────────────

@login_required
def index(request):
    projects = list(_projects(request).annotate(n_samples=Count("samples", distinct=True),
                                                n_rois=Count("roi_measurements", distinct=True)))
    for p in projects:
        p.n_annotated = sum(1 for a in Annotation.objects.filter(sample__project=p) if a.n_shapes)
        s = p.samples.filter(status=Sample.Status.READY).order_by("-created").first()
        p.thumb_id = s.id if s else None
    return render(request, "core/index.html", {"projects": projects})


@login_required
@require_POST
def project_create(request):
    data = _body(request)
    name = (data.get("name") or "").strip()
    if not name:
        return _err("Nome é obrigatório.")
    if Project.objects.filter(owner=request.user, name=name).exists():
        return _err("Já existe um projeto com esse nome.")
    p = Project.objects.create(owner=request.user, name=name, description=(data.get("description") or "").strip())
    return JsonResponse({"ok": True, "id": p.id, "url": f"/projects/{p.id}/"})


@login_required
@require_POST
def project_update(request, pid):
    p = _project(request, pid)
    data = _body(request)
    if "name" in data:
        name = (data.get("name") or "").strip()
        if not name:
            return _err("Nome é obrigatório.")
        if Project.objects.filter(owner=p.owner, name=name).exclude(id=p.id).exists():
            return _err("Já existe um projeto com esse nome.")
        p.name = name
    if "description" in data:
        p.description = (data.get("description") or "").strip()
    if "mm_per_pixel" in data:
        try:
            v = data.get("mm_per_pixel")
            p.mm_per_pixel = float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return _err("mm/pixel inválido.")
    p.save()
    return JsonResponse({"ok": True})


@login_required
@require_POST
def project_delete(request, pid):
    p = _project(request, pid)
    for s in p.samples.all():
        spectral.forget_cube(s.id)
    p.delete()   # signals.py apaga as pastas das amostras e os modelos
    return JsonResponse({"ok": True})


@login_required
def project_view(request, pid):
    """Grade de amostras do projeto (o 'Annotations' do SpectralCrops) + importacao."""
    p = _project(request, pid)
    samples = list(p.samples.select_related("annotation").annotate(n_rois=Count("roi_measurements")))
    cards, labels, notes = [], set(), set()
    for s in samples:
        ann = getattr(s, "annotation", None)
        counts = {k: len(getattr(ann, k)) if ann else 0 for k in ("boxes", "ellipses", "polygons", "points")}
        if ann and ann.image_label:
            labels.add(ann.image_label)
        if ann and ann.image_note:
            notes.add(ann.image_note)
        cards.append({"s": s, "ann": ann, "annotated": sum(counts.values()) > 0, **{f"n_{k}": v for k, v in counts.items()}})
    ready = [c for c in cards if c["s"].status == Sample.Status.READY]
    annotated = sum(1 for c in ready if c["annotated"])
    return render(request, "core/project.html", {
        "project": p, "cards": cards, "total": len(ready), "annotated": annotated,
        "progress_pct": round(annotated / len(ready) * 100, 1) if ready else 0,
        "all_labels": sorted(labels), "all_notes": sorted(notes),
        "devices": Sample.Device.choices,
        "allow_local": settings.LAMBDA_ALLOW_LOCAL_IMPORT and request.user.is_staff,
    })


# ── Importacao ────────────────────────────────────────────────────────────────
# 1) scan: recebe os arquivos (upload) ou um caminho do servidor e devolve o
#    que foi reconhecido; 2) run: importa os itens escolhidos em segundo plano.

_staging = {}
_staging_lock = threading.Lock()
STAGING_TTL = 6 * 3600


def _staging_root():
    return os.path.join(settings.MEDIA_ROOT, "_staging")


def _clean_staging():
    now = time.time()
    with _staging_lock:
        old = [k for k, v in _staging.items() if now - v["created"] > STAGING_TTL]
        for k in old:
            st = _staging.pop(k)
            if st["upload"]:
                shutil.rmtree(st["dir"], ignore_errors=True)


def _safe_relpath(rel, fallback):
    parts = [p for p in re.split(r"[\\/]+", rel or "") if p not in ("", ".", "..")]
    parts = [exports.safe_name(p, "_") for p in parts]
    return os.path.join(*parts) if parts else fallback


@login_required
@require_POST
@json_errors
def import_scan(request, pid):
    p = _project(request, pid)
    _clean_staging()
    group = request.POST.get("group", "1") == "1"
    if request.FILES:
        sdir = os.path.join(_staging_root(), uuid.uuid4().hex)
        os.makedirs(sdir)
        files = request.FILES.getlist("files")
        rels = request.POST.getlist("paths")
        for i, f in enumerate(files):
            rel = _safe_relpath(rels[i] if i < len(rels) else f.name, exports.safe_name(f.name, f"arquivo_{i}"))
            target = os.path.join(sdir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # copia em blocos: no Windows o temporario do Django some ao fechar
            # e nao pode ser movido aberto, entao mover nao e uma opcao
            with open(target, "wb") as out:
                for chunk in f.chunks(8 * 1024 * 1024):
                    out.write(chunk)
            f.close()
        roots, upload = [sdir], True
    else:
        data = _body(request)
        if not (settings.LAMBDA_ALLOW_LOCAL_IMPORT and request.user.is_staff):
            return _err("Importação de pasta local desativada para este usuário — use o upload.", 403)
        path = (data.get("path") or "").strip().strip('"')
        if not path or not os.path.exists(path):
            return _err("Caminho não encontrado no servidor.")
        group = bool(data.get("group", True))
        roots, upload, sdir = [path], False, None

    items, ignored = importers.scan(roots, group_by_prefix=group)
    sid = uuid.uuid4().hex
    with _staging_lock:
        _staging[sid] = {"owner": request.user.id, "project": p.id, "items": {it.key: it for it in items},
                         "upload": upload, "dir": sdir, "created": time.time()}
    base = roots[0]
    rel = lambda f: os.path.relpath(f, base) if os.path.isdir(base) else os.path.basename(f)
    return JsonResponse({"ok": True, "staging": sid, "upload": upload,
                         "items": [it.to_json() | {"relpath": rel(it.main_file)} for it in items],
                         "ignored": len(ignored), "ignored_examples": [rel(f) for f in ignored[:8]]})


@login_required
@require_POST
def import_run(request, pid):
    p = _project(request, pid)
    data = _body(request)
    with _staging_lock:
        st = _staging.get(data.get("staging"))
    if not st or st["owner"] != request.user.id or st["project"] != p.id:
        return _err("Varredura expirada — selecione os arquivos de novo.", 404)
    keys = data.get("keys") or list(st["items"])
    items = [st["items"][k] for k in keys if k in st["items"] and not st["items"][k].error]
    if not items:
        return _err("Nenhum item válido selecionado.")
    link = data.get("mode") == "link" and not st["upload"]
    upload = st["upload"]

    def work(report):
        created, errors = [], []
        for i, it in enumerate(items):
            report(i / len(items), f"Importando {it.name} ({i + 1}/{len(items)})")
            s = importers.import_item(p, it, link=link, move=upload)
            (created if s.status == Sample.Status.READY else errors).append({"id": s.id, "name": s.name,
                                                                               "error": s.error})
        if upload:
            with _staging_lock:
                _staging.pop(data.get("staging"), None)
            shutil.rmtree(st["dir"], ignore_errors=True)
        return {"created": created, "errors": errors}

    job = jobs.start(request.user, work, title="Importando…")
    return JsonResponse({"ok": True, "job": job})


# ── Jobs ──────────────────────────────────────────────────────────────────────

@login_required
def job_status(request, job_id):
    job = jobs.get(request.user, job_id)
    if not job:
        raise Http404("Tarefa não encontrada")
    return JsonResponse({"progress": job["progress"], "message": job["message"], "done": job["done"],
                         "error": job["error"], "result": job["result"], "download": bool(job["path"])})


@login_required
def job_download(request, job_id):
    job = jobs.get(request.user, job_id)
    if not job or not job["done"] or job["error"] or not job["path"]:
        raise Http404("Arquivo não disponível")
    jobs.pop(job_id)
    path = job["path"]
    fh = open(path, "rb")
    try:
        os.unlink(path)      # Linux: some do disco, o descritor aberto continua valendo
        delete_after = False
    except PermissionError:
        delete_after = True  # Windows: apaga depois de enviar
    resp = FileResponse(fh, as_attachment=True, filename=job["filename"])
    if delete_after:
        resp._resource_closers.append(lambda: os.path.exists(path) and os.unlink(path))
    return resp


# ── Arquivos de imagem das amostras ───────────────────────────────────────────

_png_cache = OrderedDict()
_png_lock = threading.Lock()
PNG_CACHE_SIZE = 64


def _cached_png(key, make):
    with _png_lock:
        if key in _png_cache:
            _png_cache.move_to_end(key)
            return _png_cache[key]
    ok, buf = cv2.imencode(".png", make())
    if not ok:
        raise Http404()
    data = buf.tobytes()
    with _png_lock:
        _png_cache[key] = data
        while len(_png_cache) > PNG_CACHE_SIZE:
            _png_cache.popitem(last=False)
    return data


def _png(data):
    resp = HttpResponse(data, content_type="image/png")
    resp["Cache-Control"] = "private, max-age=3600"
    return resp


@login_required
def sample_preview(request, sid):
    s = _sample(request, sid)
    path = spectral.sample_file(s, spectral.PREVIEW_NAME)
    if not os.path.exists(path):
        raise Http404()
    return FileResponse(open(path, "rb"), content_type="image/png")


@login_required
def sample_thumb(request, sid):
    s = _sample(request, sid)
    path = spectral.sample_file(s, spectral.THUMB_NAME)
    if not os.path.exists(path):
        raise Http404()
    return FileResponse(open(path, "rb"), content_type="image/jpeg")


@login_required
def sample_band(request, sid, i):
    s = _sample(request, sid)
    if not 0 <= i < s.n_bands:
        raise Http404()
    cmap = "gray" if request.GET.get("cmap") == "gray" else "spectral"
    fixed = request.GET.get("stretch") != "1"
    try:
        data = _cached_png((s.id, s.folder, "band", i, cmap, fixed), lambda: spectral.render_band(s, i, cmap, fixed))
    except FileNotFoundError:
        raise Http404()
    return _png(data)


@login_required
def sample_composite(request, sid):
    s = _sample(request, sid)
    try:
        rgb = [int(request.GET.get(k)) for k in ("r", "g", "b")]
    except (TypeError, ValueError):
        rgb = s.rgb_bands
    if not all(0 <= v < s.n_bands for v in rgb):
        raise Http404()
    try:
        data = _cached_png((s.id, s.folder, "rgb", *rgb), lambda: spectral.composite_bgr(s, rgb))
    except FileNotFoundError:
        raise Http404()
    return _png(data)


# ── Amostras ──────────────────────────────────────────────────────────────────

@login_required
@require_POST
def sample_delete(request, sid):
    s = _sample(request, sid)
    spectral.forget_cube(s.id)
    s.delete()    # signals.py remove a pasta (nunca o arquivo original vinculado)
    return JsonResponse({"ok": True})


@login_required
@require_POST
@json_errors
def sample_update(request, sid):
    """Nome, mm/pixel e bandas do preview (regera preview/miniatura)."""
    s = _sample(request, sid)
    data = _body(request)
    if "name" in data and (data["name"] or "").strip():
        s.name = data["name"].strip()
    if "mm_per_pixel" in data:
        try:
            s.mm_per_pixel = float(data["mm_per_pixel"]) if data["mm_per_pixel"] not in (None, "") else None
        except (TypeError, ValueError):
            return _err("mm/pixel inválido.")
    if "rgb_bands" in data:
        try:
            rgb = [int(v) for v in data["rgb_bands"]][:3]
        except (TypeError, ValueError):
            return _err("Bandas inválidas.")
        if len(rgb) != 3 or not all(0 <= v < s.n_bands for v in rgb):
            return _err("Bandas inválidas.")
        s.rgb_bands = rgb
        spectral.write_previews(s)
    s.save()
    spectral.write_meta(s)
    return JsonResponse({"ok": True})


@login_required
@require_POST
def annotation_image_label(request, sid):
    s = _sample(request, sid)
    ann, _ = Annotation.objects.get_or_create(sample=s)
    ann.image_label = (_body(request).get("label") or "").strip()
    ann.save()
    return JsonResponse({"ok": True, "label": ann.image_label})


@login_required
@require_POST
def annotation_image_note(request, sid):
    s = _sample(request, sid)
    ann, _ = Annotation.objects.get_or_create(sample=s)
    ann.image_note = (_body(request).get("note") or "").strip()
    ann.save()
    return JsonResponse({"ok": True, "note": ann.image_note})


@login_required
def annotate(request, sid):
    s = _sample(request, sid)
    if s.status != Sample.Status.READY:
        messages.warning(request, f"A amostra “{s.name}” não está pronta: {s.error or s.get_status_display()}")
        return redirect("project", pid=s.project_id)
    ann, _ = Annotation.objects.get_or_create(sample=s)
    p = s.project
    siblings = list(p.samples.filter(status=Sample.Status.READY).values_list("id", flat=True))
    idx = siblings.index(s.id) if s.id in siblings else 0
    spectral_cls = bool(p.model_cls) and p.model_cls.name.endswith(".joblib")
    bands = {"n": s.n_bands, "labels": [s.band_label(i) for i in range(s.n_bands)],
             "wavelengths": s.wavelengths, "names": s.band_names, "unit": s.unit,
             "monotonic": spectral.is_monotonic(s.wavelengths), "rgb": s.rgb_bands}
    return render(request, "core/annotate.html", {
        "sample": s, "project": p, "ann": ann,
        "prev_id": siblings[idx - 1] if idx > 0 else None,
        "next_id": siblings[idx + 1] if idx < len(siblings) - 1 else None,
        "current": idx + 1, "total": len(siblings),
        "models": {"seg": bool(p.model_seg), "det": bool(p.model_det), "cls": bool(p.model_cls),
                   "cls_spectral": spectral_cls},
        "bands": bands,
    })


@login_required
@require_POST
def annotation_save(request, sid):
    s = _sample(request, sid)
    data = _body(request)
    pad = lambda key, n: (list(data.get(key) or []) + [""] * n)[:n]
    boxes = [[int(round(v)) for v in b] for b in data.get("boxes") or [] if len(b) == 4]
    points = [[int(round(v)) for v in pt] for pt in data.get("points") or [] if len(pt) == 2]
    ellipses, ellipse_labels, bad_e = geometry.keep_closed(data.get("ellipses") or [],
                                                           pad("ellipse_labels", len(data.get("ellipses") or [])))
    polygons, poly_labels, bad_p = geometry.keep_closed(data.get("polygons") or [],
                                                        pad("poly_labels", len(data.get("polygons") or [])))
    to_int = lambda shapes: [[[int(round(x)), int(round(y))] for x, y in sh] for sh in shapes]
    ann, _ = Annotation.objects.get_or_create(sample=s)
    ann.boxes, ann.labels = boxes, pad("labels", len(boxes))
    ann.ellipses, ann.ellipse_labels = to_int(ellipses), ellipse_labels
    ann.polygons, ann.poly_labels = to_int(polygons), poly_labels
    ann.points, ann.point_labels = points, pad("point_labels", len(points))
    if "image_label" in data:
        ann.image_label = (data.get("image_label") or "").strip()
    ann.save()
    return JsonResponse({"ok": True, "n_boxes": len(boxes), "n_ellipses": len(ellipses),
                         "n_polygons": len(polygons), "n_points": len(points), "descartados": bad_e + bad_p})


def compute_rois(sample):
    """Recalcula (substitui) os ROIMeasurement da amostra a partir da anotacao."""
    ann = getattr(sample, "annotation", None)
    rows, counters = [], {}
    mmpp = sample.effective_mm_per_pixel
    for method, lbl, contour in geometry.annotation_rois(ann) if ann else []:
        mean, std, n = spectral.roi_spectrum(sample, contour)
        if mean is None:
            continue
        counters[method] = counters.get(method, 0) + 1
        a = geometry.area_px(contour)
        bw, bh = geometry.bbox_px(contour)
        rows.append(ROIMeasurement(
            project=sample.project, sample=sample, method=method, index=counters[method], label=lbl or "",
            contour=contour, spectrum=mean, spectrum_std=std, n_pixels=n,
            area_px=round(a, 2) if a is not None else None,
            area_mm2=round(a * mmpp ** 2, 4) if (a is not None and mmpp) else None,
            bbox_width_px=bw, bbox_height_px=bh,
            bbox_width_mm=round(bw * mmpp, 3) if (bw is not None and mmpp) else None,
            bbox_height_mm=round(bh * mmpp, 3) if (bh is not None and mmpp) else None))
    ROIMeasurement.objects.filter(sample=sample).delete()
    ROIMeasurement.objects.bulk_create(rows)
    return rows


@login_required
@require_POST
@json_errors
def roi_compute(request, sid):
    s = _sample(request, sid)
    rows = compute_rois(s)
    by_method = {}
    for r in rows:
        by_method[r.method] = by_method.get(r.method, 0) + 1
    return JsonResponse({"ok": True, "total": len(rows), "by_method": by_method})


@login_required
@require_POST
def project_rois_recompute(request, pid):
    """Recalcula os ROIs de todas as amostras (ex.: depois de mudar o mm/pixel)."""
    p = _project(request, pid)
    sample_ids = list(p.samples.filter(status=Sample.Status.READY).values_list("id", flat=True))

    def work(report):
        n = 0
        for i, sid in enumerate(sample_ids):
            report(i / max(1, len(sample_ids)), f"Medindo ROIs ({i + 1}/{len(sample_ids)})")
            s = Sample.objects.select_related("project", "annotation").get(id=sid)
            n += len(compute_rois(s))
        return {"rois": n}

    return JsonResponse({"ok": True, "job": jobs.start(request.user, work, title="Recalculando ROIs…")})


@login_required
@require_POST
@json_errors
def roi_spectrum(request, sid):
    s = _sample(request, sid)
    contour = _body(request).get("shape") or {}
    if contour.get("shape") not in ("box", "ellipse", "polygon", "point"):
        return _err("Forma inválida.")
    mean, std, n = spectral.roi_spectrum(s, contour)
    if mean is None:
        return _err("A forma não cobre nenhum pixel da imagem.")
    return JsonResponse({"ok": True, "n_pixels": n, "mean": mean, "std": std})


# ── Anotacao automatica ───────────────────────────────────────────────────────

def _preview_or_404(s):
    img = spectral.load_preview(s)
    if img is None:
        raise Http404("preview não encontrado")
    return img


@login_required
@require_POST
@json_errors
def auto_annotate(request, sid, method):
    s = _sample(request, sid)
    p = s.project
    data = _body(request)
    if method == "otsu":
        polys = autoseg.otsu_polygons(_preview_or_404(s))
        return JsonResponse({"polygons": polys, "count": len(polys)})
    if method == "boxes_to_polygons":
        boxes = data.get("boxes") or []
        if not boxes:
            return _err("Sem boxes pra converter.")
        img = _preview_or_404(s)
        labels = (list(data.get("labels") or []) + [""] * len(boxes))[:len(boxes)]
        polys, lbls = [], []
        for (x1, y1, x2, y2), lbl in zip(boxes, labels):
            pts = autoseg.polygon_from_box(img, x1, y1, x2, y2)
            if pts:
                polys.append(pts)
                lbls.append(lbl)
        return JsonResponse({"polygons": polys, "poly_labels": lbls, "count": len(polys)})
    if method == "sam":
        polys, thr = autoseg.sam_polygons(s, (data.get("polygons") or []) + (data.get("ellipses") or []))
        return JsonResponse({"polygons": polys, "count": len(polys), "threshold_deg": thr,
                             "n_bands_used": len(autoseg.sam_band_subset(s.n_bands))})
    if method == "det":
        if not p.model_det:
            return _err("Modelo de detecção não configurado (página Machine Learning).", 404)
        boxes = autoseg.yolo_boxes(p.model_det.path, _preview_or_404(s))
        return JsonResponse({"boxes": boxes, "count": len(boxes)})
    if method == "yolo":
        if not p.model_seg:
            return _err("Modelo de segmentação não configurado (página Machine Learning).", 404)
        polys = autoseg.yolo_polygons(p.model_seg.path, _preview_or_404(s))
        return JsonResponse({"polygons": polys, "count": len(polys)})
    if method == "classify":
        return _classify(s, p, data)
    raise Http404()


def _classify(s, p, data):
    if not p.model_cls:
        return _err("Modelo de classificação não configurado (página Machine Learning).", 404)
    boxes, ellipses, polygons = data.get("boxes") or [], data.get("ellipses") or [], data.get("polygons") or []
    model = ml.load_spectral_model(p.model_cls.path)
    if model:
        # classificador espectral: mede o espectro de cada forma agora e preve
        def spectra(contours):
            return [spectral.roi_spectrum(s, c)[0] for c in contours]
        box_c = [{"shape": "box", "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes]
        out = {"box_labels": ml.predict_spectra(model, s, spectra(box_c)),
               "ellipse_labels": ml.predict_spectra(model, s, spectra([{"shape": "ellipse", "points": e} for e in ellipses])),
               "poly_labels": ml.predict_spectra(model, s, spectra([{"shape": "polygon", "points": q} for q in polygons])),
               "kind": "spectral"}
        return JsonResponse(out)
    img = _preview_or_404(s)
    rect = lambda pts: (min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts))
    return JsonResponse({"box_labels": autoseg.yolo_classify_crops(p.model_cls.path, img, boxes),
                         "ellipse_labels": autoseg.yolo_classify_crops(p.model_cls.path, img, [rect(e) for e in ellipses]),
                         "poly_labels": autoseg.yolo_classify_crops(p.model_cls.path, img, [rect(q) for q in polygons]),
                         "kind": "yolo"})


# ── Analise ───────────────────────────────────────────────────────────────────

NO_CONDITION = "sem grupo"


def _fmt(v_px, v_mm, unit):
    if v_mm is not None:
        return f"{v_mm:.2f} mm{unit}"
    if v_px is not None:
        return f"{v_px:.1f} px{unit}"
    return "—"


@login_required
def analysis(request, pid):
    """Tabela por ROI + espectro medio (± desvio) por grupo experimental
    (rotulo da imagem) ou por label do ROI, na grade de bandas de referencia."""
    p = _project(request, pid)
    group_by = "roi" if request.GET.get("by") == "roi" else "image"
    rois = list(ROIMeasurement.objects.filter(project=p).select_related("sample", "sample__annotation")
                .order_by("sample__name", "method", "index"))
    X, y, groups, ref, info = ml.roi_dataset(p, label_source=group_by, unlabeled_as=NO_CONDITION)
    series = []
    if ref is not None and len(X):
        for lbl in sorted(set(y.tolist())):
            sel = X[y == lbl]
            series.append({"label": lbl, "n": int(len(sel)), "samples": int(len(set(groups[y == lbl].tolist()))),
                           "mean": [round(float(v), 3) for v in sel.mean(axis=0)],
                           "std": [round(float(v), 3) for v in (sel.std(axis=0, ddof=1) if len(sel) > 1
                                                                   else np.zeros(sel.shape[1]))]})
    rows = []
    for r in rois:
        ann = getattr(r.sample, "annotation", None)
        rows.append({"r": r, "condition": ((ann.image_label if ann else "") or "").strip() or NO_CONDITION,
                     "area": _fmt(r.area_px, r.area_mm2, "²"), "bbox_w": _fmt(r.bbox_width_px, r.bbox_width_mm, ""),
                     "bbox_h": _fmt(r.bbox_height_px, r.bbox_height_mm, "")})
    chart = {"series": series, "unit": "%",
             "x": ref["wavelengths"] if ref and ref.get("wavelengths") else None,
             "labels": ref["feature_names"] if ref else []}
    return render(request, "core/analysis.html", {
        "project": p, "rows": rows, "total": len(rois), "group_by": group_by, "chart": chart, "info": info,
        "conditions": sorted({r["condition"] for r in rows}), "methods": sorted({r["r"].get_method_display() for r in rows}),
    })


# ── Machine Learning ──────────────────────────────────────────────────────────

def _dataset_status(p, source):
    X, y, groups, ref, info = ml.roi_dataset(p, label_source=source)
    summary = ml.class_summary(y, groups) if len(X) else {}
    ready, reason = ml.readiness(summary) if summary else (False, "Nenhum ROI rotulado com espectro completo ainda.")
    cv = ml.cv_plan(summary) if ready else None
    return {"X": X, "y": y, "groups": groups, "ref": ref, "info": info, "summary": summary, "ready": ready,
            "reason": reason, "cv": cv}


@login_required
def ml_page(request, pid):
    p = _project(request, pid)
    source = "roi" if request.GET.get("source") == "roi" else "image"
    ds = _dataset_status(p, source)
    anns = list(Annotation.objects.filter(sample__project=p, sample__status=Sample.Status.READY))
    return render(request, "core/ml.html", {
        "project": p, "algorithms": ml.ALGORITHMS, "label_sources": ml.LABEL_SOURCES, "source": source,
        "dataset": {"n_samples": int(len(ds["X"])), "summary": ds["summary"], "ready": ds["ready"],
                    "reason": ds["reason"], "info": ds["info"], "cv": ds["cv"],
                    "n_features": int(ds["X"].shape[1]) if len(ds["X"]) else 0},
        "model_info": p.model_cls_info or {},
        "cls_kind": ("spectral" if p.model_cls and p.model_cls.name.endswith(".joblib")
                     else "yolo" if p.model_cls else None),
        "seg_count": sum(1 for a in anns if a.polygons or a.ellipses),
        "det_count": sum(1 for a in anns if a.boxes),
    })


@login_required
@require_POST
@json_errors
def ml_estimate(request, pid):
    p = _project(request, pid)
    data = _body(request)
    algorithm = data.get("algorithm")
    if algorithm not in ml.ALGORITHMS:
        return _err("Algoritmo desconhecido.")
    ds = _dataset_status(p, "roi" if data.get("source") == "roi" else "image")
    if not ds["ready"]:
        return _err(ds["reason"])
    params = ml.clamp_params(algorithm, data.get("params"))
    secs = ml.estimate_seconds(ds["X"], ds["y"], algorithm, params, ds["cv"][1])
    return JsonResponse({"ok": True, "estimated_seconds": round(secs, 3), "n_samples": int(len(ds["X"])),
                         "cv_mode": ds["cv"][0], "cv_folds": ds["cv"][1]})


@login_required
@require_POST
@json_errors
def ml_train(request, pid):
    p = _project(request, pid)
    data = _body(request)
    algorithm = data.get("algorithm")
    if algorithm not in ml.ALGORITHMS:
        return _err("Algoritmo desconhecido.")
    source = "roi" if data.get("source") == "roi" else "image"
    ds = _dataset_status(p, source)
    if not ds["ready"]:
        return _err(ds["reason"])
    params = ml.clamp_params(algorithm, data.get("params"))
    project_id = p.id

    def work(report):
        blob, info = ml.train(ds["X"], ds["y"], ds["groups"], ds["ref"], algorithm, params, source, report)
        proj = Project.objects.get(id=project_id)
        if proj.model_cls:
            proj.model_cls.delete(save=False)
        proj.model_cls.save(f"cls_{proj.id}_{algorithm}.joblib", ContentFile(blob), save=False)
        proj.model_cls_info = info
        proj.save(update_fields=["model_cls", "model_cls_info"])
        return {"info": info}

    return JsonResponse({"ok": True, "job": jobs.start(request.user, work, title="Treinando…")})


@login_required
@require_POST
def ml_model_upload(request, pid):
    p = _project(request, pid)
    kind = request.POST.get("type")
    if kind not in ("seg", "det", "cls"):
        return _err("type deve ser 'seg', 'det' ou 'cls'.")
    f = request.FILES.get("file")
    if not f:
        return _err("Nenhum arquivo enviado.")
    if not f.name.lower().endswith(".pt"):
        return _err("Envie um arquivo .pt (pesos do YOLO).")
    field = getattr(p, f"model_{kind}")
    if field:
        field.delete(save=False)
    field.save(f"{kind}_{p.id}_{exports.safe_name(f.name, 'modelo.pt')}", f, save=False)
    if kind == "cls":
        p.model_cls_info = {}
    p.save()
    return JsonResponse({"ok": True, "name": os.path.basename(field.name)})


@login_required
@require_POST
def ml_model_delete(request, pid):
    p = _project(request, pid)
    kind = _body(request).get("type")
    if kind not in ("seg", "det", "cls"):
        return _err("type inválido.")
    field = getattr(p, f"model_{kind}")
    if field:
        field.delete(save=False)
    if kind == "cls":
        p.model_cls_info = {}
    p.save()
    return JsonResponse({"ok": True})


@login_required
def ml_kit(request):
    """Ferramenta de treino YOLO (a mesma do SpectralCrops) pra rodar no PC com GPU."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in YOLO_KIT_FILES:
            info = zipfile.ZipInfo(f"lambda_treino_yolo/{name}")
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(os.path.join(YOLO_KIT_DIR, name), "rb") as fh:
                zf.writestr(info, fh.read())
    buf.seek(0)
    return FileResponse(buf, as_attachment=True, filename="lambda_treino_yolo.zip")


# ── Exportacoes ───────────────────────────────────────────────────────────────

@login_required
@require_POST
@json_errors
def project_export(request, pid, kind):
    p = _project(request, pid)
    data = _body(request)
    ids = [int(i) for i in data.get("ids") or []]
    if kind in ("seg", "det"):
        work, filename = exports.yolo_dataset(p, kind, ids or None, data.get("val_split", 0.2))
    elif kind == "coco":
        work, filename = exports.coco(p, ids or None)
    elif kind == "rois":
        work, filename = exports.roi_csv(p, ids or None)
    elif kind == "samples":
        work, filename = exports.samples_zip(p, ids, include_cube=bool(data.get("include_cube", True)))
    else:
        raise Http404()
    return JsonResponse({"ok": True, "job": jobs.start(request.user, work, filename=filename, title="Exportando…")})
