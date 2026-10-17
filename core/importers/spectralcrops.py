"""Pastas do SpectralCrops: uma aquisicao com raw/ e calibrated/ (ou o antigo
normalized/), cada uma com NNNnm.png, ou uma pasta do export de amostras
(365nm.png ... e, sem calibracao, 365nm_cru.png). Vale pra qualquer
equipamento que salve uma imagem por banda com o comprimento de onda no nome.
"""
import os
import re
import shutil

import cv2
import numpy as np

from .. import spectral
from .base import ImportError_, ImportItem, Importer, LoadedCube, dims_text

BAND_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*nm(_cru)?\.(png|tif|tiff)$", re.I)
SUBDIR_PRIORITY = {"calibrated": 0, "normalized": 1, "export": 2, "raw": 3}


class SpectralCropsImporter(Importer):
    device = "spectralcrops"

    def discover(self, files, claimed):
        by_acq = {}   # pasta da aquisicao -> {tipo: [arquivos de banda]}
        for f in files:
            if f in claimed or not BAND_RE.match(os.path.basename(f)):
                continue
            band_dir = os.path.dirname(f)
            kind = os.path.basename(band_dir).lower()
            acq_dir = os.path.dirname(band_dir) if kind in SUBDIR_PRIORITY else band_dir
            kind = kind if kind in SUBDIR_PRIORITY else "export"
            by_acq.setdefault(acq_dir, {}).setdefault(kind, []).append(f)

        items = []
        for acq_dir, kinds in sorted(by_acq.items()):
            all_bands = [f for fs in kinds.values() for f in fs]
            if len({_nm(f) for f in all_bands}) < 2:
                continue   # uma imagem so com "nm" no nome: deixa pro importador de imagens
            kind = min(kinds, key=lambda k: SUBDIR_PRIORITY[k])
            bands = _pick_bands(kinds[kind])
            rgb = os.path.join(acq_dir, "rgb.png")
            owned = all_bands + ([rgb] if rgb in files else [])
            item = ImportItem(key=acq_dir, name=os.path.basename(acq_dir.rstrip("\\/")) or "aquisicao",
                              device=self.device, files=owned, main_file=bands[0],
                              options={"kind": kind, "bands": bands, "rgb": rgb if rgb in files else None})
            img = spectral.imread(bands[0])
            if img is None:
                item.error = "Não foi possível ler as bandas."
            else:
                label = {"calibrated": "calibradas", "normalized": "calibradas", "export": "export", "raw": "cruas"}[kind]
                item.info = dims_text(img.shape[0], img.shape[1], len(bands), [_nm(b) for b in bands]) + f" · {label}"
            claimed.update(owned)
            items.append(item)
        return items

    def load(self, item, dest_dir, link):
        bands = item.options["bands"]
        frames, raw_bands = [], []
        for f in bands:
            img = spectral.imread(f)
            if img is None:
                raise ImportError_(f"Falha ao ler {os.path.basename(f)}.")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if frames and img.shape != frames[0].shape:
                raise ImportError_("As bandas têm tamanhos diferentes.")
            frames.append(img)
            if BAND_RE.match(os.path.basename(f)).group(2):
                raw_bands.append(_nm(f))
        dtype = frames[0].dtype
        cube = np.stack(frames, axis=2)
        np.save(os.path.join(dest_dir, spectral.CUBE_NAME), cube)
        h, w, b = cube.shape
        full_scale = 255.0 if dtype == np.uint8 else float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0

        preview = None
        if item.options.get("rgb"):
            preview = spectral.imread(item.options["rgb"], cv2.IMREAD_COLOR)
            if preview is not None and preview.shape[:2] != (h, w):
                preview = None
        if not link:   # guarda as bandas originais, com a mesma estrutura de pastas
            for f in item.files:
                target = os.path.join(dest_dir, "source", os.path.relpath(f, item.key))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                _copy(f, target, item.options.get("move", False))

        kind = item.options["kind"]
        wavelengths = [_nm(f) for f in bands]
        return LoadedCube(storage={"format": "npy", "path": spectral.CUBE_NAME, "layout": "HWB"},
                          height=h, width=w, n_bands=b, wavelengths=wavelengths,
                          band_names=[f"{x:g} nm" for x in wavelengths], value_scale=1.0 / full_scale,
                          unit="intensity" if kind == "raw" else "reflectance",
                          meta={"spectralcrops_dir": kind, "raw_bands_nm": raw_bands},
                          preview_bgr=preview)


def _copy(src, target, move):
    (shutil.move if move else shutil.copy2)(src, target)


def _nm(path):
    return float(BAND_RE.match(os.path.basename(path)).group(1))


def _pick_bands(files):
    """Uma imagem por comprimento de onda, em ordem; NNNnm.png ganha de NNNnm_cru.png."""
    best = {}
    for f in files:
        m = BAND_RE.match(os.path.basename(f))
        nm, cru = float(m.group(1)), bool(m.group(2))
        if nm not in best or (best[nm][1] and not cru):
            best[nm] = (f, cru)
    return [best[nm][0] for nm in sorted(best)]
