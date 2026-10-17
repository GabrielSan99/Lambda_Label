"""Exportacoes. Cada funcao valida/le o banco no thread do request e devolve
`work(report)` pra rodar em segundo plano (jobs.start) -- o work so le
arquivos e escreve o zip/csv temporario."""
import csv
import io
import json
import os
import re
import zipfile
from datetime import datetime

import cv2
import numpy as np

from . import geometry, jobs, spectral

_UNSAFE_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


class ExportError(Exception):
    pass


def safe_name(text, fallback):
    text = _UNSAFE_NAME_RE.sub("_", (text or "").strip()).strip(" .")
    return text or fallback


def _unique(folder, used):
    base, n = folder, 2
    while folder in used:
        folder = f"{base}_{n}"
        n += 1
    used.add(folder)
    return folder


def _sample_info(s):
    return {"amostra_id": s.id, "nome": s.name, "equipamento": s.device, "largura": s.width, "altura": s.height,
            "n_bandas": s.n_bands, "comprimentos_onda_nm": s.wavelengths, "nomes_bandas": s.band_names,
            "unidade": s.unit, "mm_por_pixel": s.effective_mm_per_pixel, "origem": s.source_path,
            "importada_em": s.created.strftime("%Y-%m-%d %H:%M:%S")}


# ── Dataset YOLO (segmentacao / deteccao) ─────────────────────────────────────

def yolo_dataset(project, task, ids=None, val_split=0.2):
    from .models import Annotation
    if task not in ("seg", "det"):
        raise ExportError("task deve ser 'seg' ou 'det'")
    val_split = max(0.05, min(0.5, float(val_split)))
    anns = Annotation.objects.filter(sample__project=project, sample__status="ready").select_related("sample")
    if ids:
        anns = anns.filter(sample_id__in=ids)
    items, all_labels = [], set()
    for a in anns:
        if task == "seg":
            # segmentacao: elipse + poligono (os dois sao contorno fechado)
            shapes, labels, _ = geometry.keep_closed(list(a.ellipses) + list(a.polygons),
                                                     list(a.ellipse_labels) + [""] * (len(a.ellipses) - len(a.ellipse_labels))
                                                     + list(a.poly_labels))
        else:
            shapes, labels = a.boxes, list(a.labels) + [""] * (len(a.boxes) - len(a.labels))
        if not shapes:
            continue
        img_path = spectral.sample_file(a.sample, spectral.PREVIEW_NAME)
        if not os.path.exists(img_path):
            continue
        if task == "det":
            all_labels.update(l for l in labels if l)
        items.append((a.sample, img_path, shapes, labels))
    if not items:
        raise ExportError(f"Nenhuma anotação com {'polígonos/elipses' if task == 'seg' else 'caixas'} nas amostras escolhidas.")

    # segmentacao binaria (amostra x fundo), como no SpectralCrops
    label_map = {"amostra": 0} if task == "seg" else (
        {l: i for i, l in enumerate(sorted(all_labels))} if all_labels else {"objeto": 0})
    rng = np.random.RandomState(42)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, round(len(order) * val_split)) if len(order) > 1 else 0
    val_idx = set(order[:n_val])
    names_yaml = "\n".join(f"  {i}: {n}" for n, i in sorted(label_map.items(), key=lambda kv: kv[1]))
    data_yaml = f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names_yaml}\n"

    def work(report):
        def write(fh):
            used = set()
            with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, (s, img_path, shapes, labels) in enumerate(items):
                    split = "val" if i in val_idx else "train"
                    stem = _unique(re.sub(r"[^A-Za-z0-9_.-]+", "_", s.name) or f"amostra_{s.id}", used)
                    w, h = s.width, s.height
                    lines = []
                    for shape, lbl in zip(shapes, labels):
                        if task == "seg":
                            lines.append("0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in shape))
                        else:
                            x1, y1, x2, y2 = shape
                            lines.append(f"{label_map.get(lbl, 0)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    zf.write(img_path, f"images/{split}/{stem}.png")
                    zf.writestr(f"labels/{split}/{stem}.txt", "\n".join(lines))
                    report(0.95 * (i + 1) / len(items))
                zf.writestr("data.yaml", data_yaml)
        return jobs.temp_file(".zip", write)

    return work, f"dataset_{task}_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip"


# ── COCO (interoperabilidade com CVAT, Label Studio, detectron2, mmdet...) ────

def coco(project, ids=None):
    from .models import Annotation
    anns = Annotation.objects.filter(sample__project=project, sample__status="ready").select_related("sample")
    if ids:
        anns = anns.filter(sample_id__in=ids)
    anns = [a for a in anns if a.boxes or a.polygons or a.ellipses]
    if not anns:
        raise ExportError("Nenhuma das amostras escolhidas tem caixas, polígonos ou elipses.")
    categories = {}

    def cat(lbl):
        lbl = (lbl or "").strip() or "objeto"
        if lbl not in categories:
            categories[lbl] = len(categories) + 1
        return categories[lbl]

    images, annotations, files, used = [], [], [], set()
    for img_id, a in enumerate(anns, start=1):
        s = a.sample
        fname = _unique(safe_name(s.name, f"amostra_{s.id}") + ".png", used)
        images.append({"id": img_id, "file_name": fname, "width": s.width, "height": s.height,
                       "lambda_label": _sample_info(s) | {"rotulo_imagem": a.image_label, "observacao": a.image_note}})
        files.append((spectral.sample_file(s, spectral.PREVIEW_NAME), f"images/{fname}"))
        pad = lambda l, n: list(l or []) + [""] * max(0, n - len(l or []))
        polys = ([(p, l) for p, l in zip(a.polygons, pad(a.poly_labels, len(a.polygons)))]
                 + [(p, l) for p, l in zip(a.ellipses, pad(a.ellipse_labels, len(a.ellipses)))])
        for pts, lbl in polys:
            pts = geometry.closed_polygon(pts)
            if not pts:
                continue
            arr = np.array(pts, dtype=np.float32)
            x0, y0 = arr.min(axis=0)
            x1, y1 = arr.max(axis=0)
            annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": cat(lbl or a.image_label),
                                "segmentation": [[round(float(v), 2) for xy in pts for v in xy]],
                                "area": float(abs(cv2.contourArea(arr))),
                                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "iscrowd": 0})
        for (x1, y1, x2, y2), lbl in zip(a.boxes, pad(a.labels, len(a.boxes))):
            annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": cat(lbl or a.image_label),
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": float((x2 - x1) * (y2 - y1)),
                                "segmentation": [], "iscrowd": 0})
    doc = {"info": {"description": f"Lambda Label — {project.name}", "date_created": datetime.now().isoformat()},
           "images": images, "annotations": annotations,
           "categories": [{"id": i, "name": n, "supercategory": ""} for n, i in categories.items()]}

    def work(report):
        def write(fh):
            with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, (src, dst) in enumerate(files):
                    if os.path.exists(src):
                        zf.write(src, dst)
                    report(0.9 * (i + 1) / len(files))
                zf.writestr("annotations.json", json.dumps(doc, ensure_ascii=False, indent=2))
        return jobs.temp_file(".zip", write)

    return work, f"coco_{safe_name(project.name, 'projeto')}_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip"


# ── CSV dos espectros por ROI ─────────────────────────────────────────────────

def roi_csv(project, ids=None):
    from .models import ROIMeasurement
    rois = (ROIMeasurement.objects.filter(project=project).select_related("sample", "sample__annotation")
            .order_by("sample__name", "method", "index"))
    if ids:
        rois = rois.filter(sample_id__in=ids)
    rois = list(rois)
    if not rois:
        raise ExportError("Nenhum ROI medido nas amostras escolhidas — salve as anotações no editor primeiro.")
    # colunas = uniao dos rotulos de banda (amostras de equipamentos diferentes tem bandas diferentes)
    columns, seen = [], set()
    for r in rois:
        for i in range(r.sample.n_bands):
            lbl = r.sample.band_label(i)
            if lbl not in seen:
                seen.add(lbl)
                columns.append(lbl)
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["Amostra", "Equipamento", "Rótulo da imagem", "Método", "ROI", "Label", "Pixels",
                 "Área (px²)", "Área (mm²)", "Largura bbox (px)", "Altura bbox (px)", "Largura bbox (mm)",
                 "Altura bbox (mm)", "Medida"] + columns)
    for r in rois:
        ann = getattr(r.sample, "annotation", None)
        unit = "reflectância_%" if r.sample.unit == "reflectance" else "intensidade_%"
        base = [r.sample.name, r.sample.device, ann.image_label if ann else "", r.get_method_display(), r.index,
                r.label, r.n_pixels, r.area_px, r.area_mm2, r.bbox_width_px, r.bbox_height_px,
                r.bbox_width_mm, r.bbox_height_mm]
        for kind, values in ((unit, r.spectrum), ("desvio_" + unit, r.spectrum_std)):
            by_lbl = {r.sample.band_label(i): v for i, v in enumerate(values or [])}
            wr.writerow(base + [kind] + [("" if by_lbl.get(c) is None else by_lbl[c]) for c in columns])
    text = buf.getvalue()

    def work(report):
        return jobs.temp_file(".csv", lambda fh: fh.write(text.encode("utf-8-sig")))

    return work, f"roi_espectros_{safe_name(project.name, 'projeto')}.csv"


# ── Amostras pra classificacao (mesmo esquema do export do SpectralCrops) ─────

SAMPLES_README = """Amostras -- Lambda Label
========================

anotacoes.json         ÍNDICE ÚNICO: projeto, imagens, caminhos, bandas e amostras.
<imagem>/rgb.png       composição colorida usada na anotação (só visualização).
<imagem>/cube.npy      {cube_txt}
                       np.load(...) -> (altura, largura, bandas). Valor * escala_valor
                       (em anotacoes.json) = fração 0..1 (reflectância ou intensidade).

anotacoes.json -- caminhos RELATIVOS a esta pasta:
  exportado_em, projeto
  imagens: lista --
    imagem, amostra_id, equipamento, largura, altura, n_bandas,
    comprimentos_onda_nm, nomes_bandas, unidade, escala_valor, mm_por_pixel,
    rotulo_imagem, observacao, pasta, rgb, cubo
    amostras: lista numerada em linhas (cima -> baixo, esquerda -> direita).
      Cada amostra junta os contornos da mesma semente/objeto (o centro de um
      dentro do outro). label: o do 1º contorno com label (polígono > elipse >
      box > ponto); sem nenhum, o rótulo da imagem.
      contornos: tipo (polygon | ellipse | box | point), indice, label,
                 label_origem, pontos [[x, y], ...] | caixa [x1, y1, x2, y2] | ponto [x, y]

Treino x validação: divida por IMAGEM (amostra_id), não por objeto -- objetos
da mesma imagem compartilham luz, lote e posição.

Exportado: {n_images} imagem(ns), {n_samples} amostra(s), {n_contours} contorno(s).
Labels:
{counts}
"""


def samples_zip(project, ids, include_cube=True):
    from .models import Annotation
    anns = [a for a in Annotation.objects.filter(sample__project=project, sample_id__in=ids or [],
                                                 sample__status="ready").select_related("sample")
            .order_by("sample__name") if a.n_shapes]
    if not anns:
        raise ExportError("Nenhuma das amostras escolhidas tem anotação.")
    used = set()
    jobs_list = [(a.sample, _unique(safe_name(a.sample.name, f"amostra_{a.sample.id}"), used),
                  a.image_label or "", a.image_note or "", geometry.group_samples(a)) for a in anns]

    def work(report):
        counts, totals = {}, {"samples": 0, "contours": 0}

        def write(fh):
            images = []
            with zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for k, (s, folder, img_label, note, groups) in enumerate(jobs_list):
                    report(0.95 * k / len(jobs_list), f"Exportando {s.name}")
                    zf.write(spectral.sample_file(s, spectral.PREVIEW_NAME), f"{folder}/rgb.png")
                    cube_rel = None
                    if include_cube:
                        cube_rel = f"{folder}/cube.npy"
                        arr = spectral.open_cube(s)
                        with zf.open(cube_rel, "w", force_zip64=True) as out:
                            np.lib.format.write_array(out, arr, allow_pickle=False)
                    entry = _sample_info(s) | {"imagem": s.name, "escala_valor": s.value_scale,
                                               "rotulo_imagem": img_label, "observacao": note, "pasta": folder,
                                               "rgb": f"{folder}/rgb.png", "cubo": cube_rel, "amostras": groups}
                    images.append(entry)
                    for g in groups:
                        key = g["label"] or "(sem label)"
                        counts[key] = counts.get(key, 0) + 1
                        totals["contours"] += len(g["contornos"])
                    totals["samples"] += len(groups)
                index = {"exportado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "projeto": {"id": project.id, "nome": project.name, "descricao": project.description},
                         "imagens": images}
                zf.writestr("anotacoes.json", json.dumps(index, ensure_ascii=False, indent=2))
                zf.writestr("LEIA-ME.txt", SAMPLES_README.format(
                    cube_txt="cubo espectral completo (formato e escala originais)" if include_cube
                    else "(não incluído neste export)",
                    n_images=len(jobs_list), n_samples=totals["samples"], n_contours=totals["contours"],
                    counts="\n".join(f"  {c}: {n}" for c, n in sorted(counts.items()))))
        return jobs.temp_file(".zip", write)

    return work, f"amostras_{safe_name(project.name, 'projeto')}_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip"
