"""Imagens comuns (PNG/JPG/TIFF/BMP) e conjuntos de imagens da mesma cena.

- Uma imagem colorida vira uma amostra de 3 bandas (R, G, B); cinza, 1 banda;
  TIFF multipagina, uma banda por pagina (8 ou 16 bits).
- "Agrupar por prefixo": arquivos AMOSTRA_RGB.jpg + AMOSTRA_CHL.jpg +
  AMOSTRA_ARIIDX.jpg (ex.: PhenoVation SeedReporter) viram UMA amostra com as
  bandas de todos, nomeadas pelo sufixo. Sufixo tipo "660nm" vira comprimento
  de onda.
"""
import os
import re

import cv2
import numpy as np

from .. import spectral
from .base import ImportError_, ImportItem, Importer, LoadedCube, dims_text, place_file, stem_of

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
NM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*nm$", re.I)


def read_pages(path):
    """[ndarray] -- uma por pagina (TIFF multipagina) ou so uma."""
    if path.lower().endswith((".tif", ".tiff")):
        pages = spectral.imreadmulti(path)
        if pages:
            return pages
    img = spectral.imread(path)
    return [img] if img is not None else []


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            channels = len(im.getbands())
            return im.height, im.width, channels * getattr(im, "n_frames", 1)
    except Exception:
        pages = read_pages(path)
        if not pages:
            return None
        p = pages[0]
        return p.shape[0], p.shape[1], sum(1 if x.ndim == 2 else min(3, x.shape[2]) for x in pages)


class ImageImporter(Importer):
    device = "image"

    def __init__(self, group_by_prefix=True):
        self.group_by_prefix = group_by_prefix

    def discover(self, files, claimed):
        imgs = [f for f in files if f not in claimed and f.lower().endswith(IMAGE_EXTS)]
        groups = {}
        for f in imgs:
            stem = stem_of(f)
            prefix = stem.rsplit("_", 1)[0] if self.group_by_prefix and "_" in stem else stem
            groups.setdefault((os.path.dirname(f), prefix), []).append(f)

        items = []
        for (folder, prefix), members in sorted(groups.items()):
            members.sort()
            singles = [[m] for m in members] if len(members) < 2 else [members]
            for group in singles:
                name = prefix if len(group) > 1 else stem_of(group[0])
                item = ImportItem(key="|".join(group), name=name, device=self.device, files=group,
                                  main_file=group[0])
                sizes = [_image_size(m) for m in group]
                if any(s is None for s in sizes):
                    item.error = "Imagem ilegível."
                elif len({s[:2] for s in sizes}) > 1:
                    item.error = "Imagens do grupo têm tamanhos diferentes — importe separadas."
                else:
                    h, w = sizes[0][:2]
                    item.info = dims_text(h, w, sum(s[2] for s in sizes))
                    if len(group) > 1:
                        item.info += " · " + ", ".join(_suffix(m, prefix) for m in group)
                claimed.update(group)
                items.append(item)
        return items

    def load(self, item, dest_dir, link):
        prefix = item.name
        bands, names, wavelengths, preview, color_blocks = [], [], [], None, []
        for f in item.files:
            pages = read_pages(f)
            if not pages:
                raise ImportError_(f"Falha ao ler {os.path.basename(f)}.")
            tag = _suffix(f, prefix) if len(item.files) > 1 else ""
            for p_i, page in enumerate(pages):
                ptag = tag if len(pages) == 1 else f"{tag or 'pág'} {p_i + 1}".strip()
                if page.ndim == 3:
                    if page.shape[2] == 4:
                        page = page[:, :, :3]
                    color_blocks.append((ptag, len(bands), page))
                    for c, cname in ((2, "R"), (1, "G"), (0, "B")):
                        bands.append(page[:, :, c])
                        names.append(f"{ptag} {cname}".strip())
                        wavelengths.append(None)
                else:
                    bands.append(page)
                    names.append(ptag or "cinza")
                    m = NM_RE.search(ptag)
                    wavelengths.append(float(m.group(1)) if m else None)
        if len({b.shape for b in bands}) > 1:
            raise ImportError_("As bandas têm tamanhos diferentes.")

        dtypes = {b.dtype for b in bands}
        if dtypes == {np.dtype(np.uint8)}:
            cube, scale = np.stack(bands, axis=2), 1.0 / 255.0
        else:
            cube = np.stack([_to_fraction(b) for b in bands], axis=2).astype(np.float32)
            scale = 1.0
        np.save(os.path.join(dest_dir, spectral.CUBE_NAME), cube)
        if not link:
            for f in item.files:
                place_file(f, dest_dir, link=False, move=item.options.get("move", False))
        h, w, b = cube.shape
        if not all(x is not None for x in wavelengths):
            wavelengths = []
        # preview = a propria foto colorida (sem stretch), preferindo a do sufixo
        # "RGB" num grupo (SeedReporter: X_RGB.jpg ao lado de X_CHL.jpg/X_ARIIDX.jpg)
        rgb = None
        if color_blocks:
            tag, start, page = next((blk for blk in color_blocks if "RGB" in blk[0].upper()), color_blocks[0])
            rgb = [start, start + 1, start + 2]
            if page.dtype == np.uint8:
                preview = page
        return LoadedCube(storage={"format": "npy", "path": spectral.CUBE_NAME, "layout": "HWB"},
                          height=h, width=w, n_bands=b, wavelengths=wavelengths, band_names=names,
                          value_scale=scale, unit="intensity", rgb_bands=rgb,
                          meta={"files": [os.path.basename(f) for f in item.files]}, preview_bgr=preview)


def _suffix(path, prefix):
    stem = stem_of(path)
    return stem[len(prefix) + 1:] if stem.startswith(prefix + "_") else stem


def _to_fraction(band):
    if np.issubdtype(band.dtype, np.integer):
        return band.astype(np.float32) / float(np.iinfo(band.dtype).max)
    return band.astype(np.float32)
