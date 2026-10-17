"""Testes de ponta a ponta com dados sinteticos (nao dependem de arquivos reais).

    python manage.py test core
"""
import io
import json
import os
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase, override_settings

from . import autoseg, geometry, importers, ml, spectral
from .models import Annotation, Project, ROIMeasurement, Sample

H, W, B = 120, 160, 12
WL = np.linspace(400, 950, B)
SEEDS = [(40, 30), (80, 30), (120, 30), (40, 85), (80, 85), (120, 85)]   # (x, y) centros
R = 11


def spectrum_of(kind):
    return np.linspace(0.2, 0.6, B) if kind == "A" else np.linspace(0.6, 0.2, B)


def scene(kind, seed=0):
    rng = np.random.RandomState(seed)
    cube = np.full((H, W, B), 0.05, np.float32) + rng.normal(0, 0.005, (H, W, B)).astype(np.float32)
    yy, xx = np.mgrid[:H, :W]
    for cx, cy in SEEDS:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
        cube[m] = spectrum_of(kind) + rng.normal(0, 0.01, (m.sum(), B))
    return cube


def write_envi(path, cube, interleave="bil"):
    h, w, b = cube.shape
    data = {"bil": cube.transpose(0, 2, 1), "bip": cube, "bsq": cube.transpose(2, 0, 1)}[interleave]
    np.ascontiguousarray(data, dtype="<f4").tofile(path)
    with open(path + ".hdr", "w") as fh:
        fh.write(f"ENVI\ninterleave = {interleave}\ndata type = 4\nlines = {h}\nsamples = {w}\nbands = {b}\n"
                 f"byte order = 0\nheader offset = 0\nwavelength = {{{', '.join(f'{x:.2f}' for x in WL)}}}\n"
                 f"wavelength units = nm\nreflectance scale factor = 1.0\n")


def circle(cx, cy, r=R - 2, n=24):
    return [[int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


class Base(TransactionTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lambda_test_")
        self.media = os.path.join(self.tmp, "media")
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(self.src)
        self.override = override_settings(MEDIA_ROOT=self.media)
        self.override.enable()
        self.user = get_user_model().objects.create_user("u", password="p", is_staff=True)
        self.project = Project.objects.create(owner=self.user, name="P")

    def tearDown(self):
        for s in Sample.objects.all():
            spectral.forget_cube(s.id)
        self.override.disable()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_sources(self):
        write_envi(os.path.join(self.src, "envi_A1.bil"), scene("A", 1), "bil")
        write_envi(os.path.join(self.src, "envi_B1.bsq"), scene("B", 2), "bsq")
        np.save(os.path.join(self.src, "cubo.npy"), (scene("A", 3) * 1000).astype(np.uint16).transpose(2, 0, 1))
        with open(os.path.join(self.src, "cubo.json"), "w") as fh:
            json.dump({"wavelengths": WL.tolist(), "value_scale": 0.001, "unit": "reflectance", "layout": "BHW",
                       "mm_per_pixel": 0.05}, fh)
        rgb = (scene("A", 4)[:, :, [2, 6, 10]] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.src, "folha_RGB.png"), rgb)
        cv2.imwrite(os.path.join(self.src, "folha_CHL.png"), rgb[:, :, 1])
        acq = os.path.join(self.src, "2026-10-03_acq-1")
        for sub in ("raw", "calibrated"):
            os.makedirs(os.path.join(acq, sub))
            for nm in (365, 460, 520, 660, 850):
                cv2.imwrite(os.path.join(acq, sub, f"{nm}nm.png"), rgb[:, :, 0])
        cv2.imwrite(os.path.join(acq, "rgb.png"), rgb)
        with open(os.path.join(self.src, "leia.txt"), "w") as fh:
            fh.write("ignorar")


class ImportTests(Base):
    def test_scan_and_import_all_formats(self):
        self.make_sources()
        items, ignored = importers.scan([self.src])
        by_dev = {}
        for it in items:
            by_dev.setdefault(it.device, []).append(it)
            self.assertEqual(it.error, "", it.name)
        self.assertEqual(len(by_dev["envi"]), 2)
        self.assertEqual(len(by_dev["npy"]), 1)
        self.assertEqual(len(by_dev["image"]), 1)            # folha_RGB + folha_CHL agrupadas
        self.assertEqual(len(by_dev["spectralcrops"]), 1)    # raw/ + calibrated/ = 1 aquisicao
        self.assertTrue(any(f.endswith("leia.txt") for f in ignored))

        samples = {it.name: importers.import_item(self.project, it, link=True) for it in items}
        for name, s in samples.items():
            self.assertEqual(s.status, Sample.Status.READY, f"{name}: {s.error}")
            self.assertTrue(os.path.exists(spectral.sample_file(s, spectral.PREVIEW_NAME)))
            self.assertEqual((s.height, s.width), (H, W))
        envi = samples["envi_A1"]
        self.assertEqual(envi.n_bands, B)
        self.assertTrue(envi.linked)
        self.assertEqual(os.path.abspath(envi.storage["path"]), os.path.join(self.src, "envi_A1.bil"))
        npy = samples["cubo"]
        self.assertEqual(npy.n_bands, B)
        self.assertAlmostEqual(npy.mm_per_pixel, 0.05)
        self.assertEqual(samples["folha"].n_bands, 4)          # R, G, B + CHL
        sc = samples["2026-10-03_acq-1"]
        self.assertEqual(sc.wavelengths, [365.0, 460.0, 520.0, 660.0, 850.0])
        self.assertEqual(sc.meta["spectralcrops_dir"], "calibrated")

    def test_interleaves_read_identically(self):
        cube = scene("A", 7)
        for il in ("bil", "bip", "bsq"):
            write_envi(os.path.join(self.src, f"c_{il}.raw"), cube, il)
        items, _ = importers.scan([self.src])
        for it in items:
            s = importers.import_item(self.project, it, link=True)
            np.testing.assert_allclose(spectral.read_bands(s), cube, atol=1e-6, err_msg=it.name)
            np.testing.assert_allclose(spectral.band(s, 5), cube[:, :, 5], atol=1e-6)

    def test_bad_envi_reports_error(self):
        with open(os.path.join(self.src, "x.bil"), "wb") as fh:
            fh.write(b"\0" * 100)
        with open(os.path.join(self.src, "x.bil.hdr"), "w") as fh:
            fh.write("ENVI\ninterleave = bil\ndata type = 4\nlines = 10\nsamples = 10\nbands = 10\n")
        items, _ = importers.scan([self.src])
        s = importers.import_item(self.project, items[0], link=True)
        self.assertEqual(s.status, Sample.Status.ERROR)
        self.assertIn("menor que o esperado", s.error)


class SpectralTests(Base):
    def setUp(self):
        super().setUp()
        write_envi(os.path.join(self.src, "a.bil"), scene("A", 1))
        items, _ = importers.scan([self.src])
        self.s = importers.import_item(self.project, items[0], link=True)

    def test_roi_spectrum_matches_scene(self):
        mean, std, n = spectral.roi_spectrum(self.s, {"shape": "polygon", "points": circle(*SEEDS[0])})
        self.assertGreater(n, 100)
        np.testing.assert_allclose(np.array(mean) / 100, spectrum_of("A"), atol=0.01)
        box_mean, _, _ = spectral.roi_spectrum(self.s, {"shape": "box", "x1": 36, "y1": 26, "x2": 44, "y2": 34})
        np.testing.assert_allclose(np.array(box_mean) / 100, spectrum_of("A"), atol=0.01)

    def test_sam_finds_other_seeds(self):
        polys, thr = autoseg.sam_polygons(self.s, [circle(*SEEDS[0], r=R)])
        self.assertEqual(len(polys), len(SEEDS) - 1)
        self.assertIsNotNone(thr)

    def test_otsu_and_box_to_polygon(self):
        img = spectral.load_preview(self.s)
        self.assertEqual(len(autoseg.otsu_polygons(img)), len(SEEDS))
        cx, cy = SEEDS[1]
        pts = autoseg.polygon_from_box(img, cx - 16, cy - 16, cx + 16, cy + 16)
        self.assertIsNotNone(pts)

    def test_group_samples(self):
        ann = Annotation.objects.get(sample=self.s)
        ann.polygons = [circle(*SEEDS[0])]
        ann.boxes = [[SEEDS[0][0] - 10, SEEDS[0][1] - 10, SEEDS[0][0] + 10, SEEDS[0][1] + 10]]
        ann.points = [list(SEEDS[3])]
        ann.poly_labels, ann.labels, ann.point_labels = ["x"], [""], ["p"]
        groups = geometry.group_samples(ann)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["label"], "x")
        self.assertEqual(len(groups[0]["contornos"]), 2)


@override_settings(LAMBDA_ALLOW_LOCAL_IMPORT=True)
class HttpFlowTests(Base):
    def wait(self, job):
        for _ in range(300):
            st = self.client.get(f"/jobs/{job}/").json()
            if st["done"]:
                self.assertIsNone(st["error"], st["error"])
                return st
            time.sleep(0.1)
        self.fail("job não terminou")

    def post(self, url, data=None):
        return self.client.post(url, json.dumps(data or {}), content_type="application/json")

    def test_full_flow(self):
        c = self.client
        self.assertEqual(c.get("/").status_code, 302)              # exige login
        c.login(username="u", password="p")
        self.assertEqual(c.get("/").status_code, 200)
        pid = self.project.id

        # upload de 4 cubos ENVI (2 por classe) pelo navegador
        files, paths = [], []
        for kind, k in (("A", 1), ("A", 2), ("B", 3), ("B", 4)):
            base = os.path.join(self.src, f"{kind}{k}.bil")
            write_envi(base, scene(kind, k))
            for f in (base, base + ".hdr"):
                files.append(SimpleUploadedFile(os.path.basename(f), open(f, "rb").read()))
                paths.append(f"lote/{os.path.basename(f)}")
        r = c.post(f"/projects/{pid}/import/scan/", {"files": files, "paths": paths, "group": "1"})
        self.assertEqual(r.status_code, 200, r.content)
        scan = r.json()
        self.assertEqual(len(scan["items"]), 4)
        r = self.post(f"/projects/{pid}/import/run/", {"staging": scan["staging"], "mode": "copy"})
        st = self.wait(r.json()["job"])
        self.assertEqual(len(st["result"]["created"]), 4)
        samples = list(Sample.objects.filter(project=self.project).order_by("name"))
        self.assertTrue(all(s.status == "ready" and not s.linked for s in samples))
        self.assertEqual(c.get(f"/projects/{pid}/").status_code, 200)

        # pasta local (staff) -- so a varredura
        r = self.post(f"/projects/{pid}/import/scan/", {"path": self.src})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["items"]), 4)

        # anota: rotulo da imagem = classe, 4 poligonos por amostra
        for s in samples:
            self.assertEqual(c.get(f"/samples/{s.id}/").status_code, 200)
            for url in (f"/samples/{s.id}/preview.png", f"/samples/{s.id}/thumb.jpg", f"/samples/{s.id}/band/3.png",
                        f"/samples/{s.id}/composite.png?r=10&g=5&b=1"):
                self.assertEqual(c.get(url).status_code, 200, url)
            self.post(f"/samples/{s.id}/label/", {"label": "verde" if s.name.startswith("A") else "normal"})
            r = self.post(f"/samples/{s.id}/save/", {"polygons": [circle(*p) for p in SEEDS[:4]], "poly_labels": [],
                                                     "boxes": [[10, 10, 20, 20]], "points": [[5, 5]]})
            self.assertTrue(r.json()["ok"])
            r = self.post(f"/samples/{s.id}/rois/compute/")
            self.assertEqual(r.json()["total"], 6)
        r = self.post(f"/samples/{samples[0].id}/spectrum/", {"shape": {"shape": "polygon", "points": circle(*SEEDS[0])}})
        self.assertEqual(len(r.json()["mean"]), B)
        r = self.post(f"/samples/{samples[0].id}/auto/sam/", {"polygons": [circle(*SEEDS[0], r=R)]})
        self.assertEqual(r.json()["count"], len(SEEDS) - 1)
        r = self.post(f"/samples/{samples[0].id}/auto/otsu/")
        self.assertEqual(r.json()["count"], len(SEEDS))
        r = self.post(f"/samples/{samples[0].id}/auto/det/")
        self.assertEqual(r.status_code, 404)                         # sem modelo YOLO: mensagem, nao 500

        # analise
        r = c.get(f"/projects/{pid}/analysis/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.context["chart"]["series"]), 2)

        # treino do classificador espectral (box/ponto de fundo entram como ROI da classe da imagem;
        # mesmo assim os espectros das sementes separam as classes)
        self.assertEqual(c.get(f"/projects/{pid}/ml/").status_code, 200)
        r = self.post(f"/projects/{pid}/ml/estimate/", {"algorithm": "random_forest", "params": {"n_estimators": 20}})
        self.assertTrue(r.json()["ok"], r.content)
        self.assertEqual(r.json()["cv_mode"], "group")
        r = self.post(f"/projects/{pid}/ml/train/", {"algorithm": "logistic_regression", "params": {}})
        self.wait(r.json()["job"])
        self.project.refresh_from_db()
        self.assertTrue(self.project.model_cls.name.endswith(".joblib"))
        self.assertGreaterEqual(self.project.model_cls_info["cv_accuracy_mean"], 0.5)
        r = self.post(f"/samples/{samples[-1].id}/auto/classify/", {"polygons": [circle(*SEEDS[5])]})
        self.assertEqual(r.json()["kind"], "spectral")
        self.assertEqual(r.json()["poly_labels"], ["normal"])
        self.assertEqual(c.get(f"/projects/{pid}/ml/").status_code, 200)

        # exportacoes
        for kind, member in (("seg", "data.yaml"), ("det", "data.yaml"), ("coco", "annotations.json"),
                             ("samples", "anotacoes.json")):
            r = self.post(f"/projects/{pid}/export/{kind}/", {"ids": [s.id for s in samples]})
            job = r.json()["job"]
            self.wait(job)
            resp = c.get(f"/jobs/{job}/download/")
            z = zipfile.ZipFile(io.BytesIO(b"".join(resp.streaming_content)))
            self.assertIn(member, z.namelist(), kind)
            if kind == "samples":
                idx = json.loads(z.read("anotacoes.json"))
                self.assertEqual(len(idx["imagens"]), 4)
                cube = np.load(io.BytesIO(z.read(idx["imagens"][0]["cubo"])))
                self.assertEqual(cube.shape, (H, W, B))
            resp.close()
        r = self.post(f"/projects/{pid}/export/rois/", {})
        job = r.json()["job"]
        self.wait(job)
        resp = c.get(f"/jobs/{job}/download/")
        text = b"".join(resp.streaming_content).decode("utf-8-sig")
        resp.close()
        self.assertEqual(len(text.strip().splitlines()), 1 + 2 * ROIMeasurement.objects.count())

        # exclusao limpa as pastas
        folder = spectral.sample_dir(samples[0])
        self.assertTrue(os.path.isdir(folder))
        self.post(f"/samples/{samples[0].id}/delete/")
        self.assertFalse(os.path.isdir(folder))

    def test_other_user_cannot_access(self):
        other = get_user_model().objects.create_user("o", password="p")
        write_envi(os.path.join(self.src, "a.bil"), scene("A", 1))
        items, _ = importers.scan([self.src])
        s = importers.import_item(self.project, items[0], link=True)
        self.client.login(username="o", password="p")
        self.assertEqual(self.client.get(f"/projects/{self.project.id}/").status_code, 404)
        self.assertEqual(self.client.get(f"/samples/{s.id}/").status_code, 404)
        r = self.post(f"/projects/{self.project.id}/import/scan/", {"path": self.src})
        self.assertEqual(r.status_code, 404)
