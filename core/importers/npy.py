"""Formato generico de integracao: um cubo NumPy (.npy) + um .json opcional
com o mesmo nome. E o caminho pra trazer dado de QUALQUER equipamento que
ainda nao tenha importador proprio -- basta salvar o cubo do lado de la:

    np.save("amostra01.npy", cubo)            # (altura, largura, bandas)
    json.dump({"wavelengths": [450, 550, 650, ...],   # nm, um por banda (opcional)
               "band_names": ["azul", ...],           # opcional
               "value_scale": 0.01,                   # valor * escala = fracao 0..1 (opcional)
               "unit": "reflectance",                 # ou "intensity" (opcional)
               "mm_per_pixel": 0.05,                  # opcional
               "layout": "HWB"},                      # ou "BHW" se as bandas vierem primeiro
              open("amostra01.json", "w"))
"""
import json
import os

import numpy as np

from .base import ImportError_, ImportItem, Importer, LoadedCube, dims_text, place_file, stem_of


def _sidecar(npy_path):
    path = os.path.splitext(npy_path)[0] + ".json"
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return path, json.load(fh)
        except (OSError, ValueError):
            return path, {}
    return None, {}


def _layout(shape, meta):
    """Sem "layout" no .json: a dimensao das bandas e a menor -- se a ultima ja
    e a menor, (H, W, B); se a primeira e, (B, H, W). Recorte com mais bandas
    que pixels de lado fica ambiguo: declare o layout no .json."""
    if meta.get("layout") in ("HWB", "BHW"):
        return meta["layout"]
    if shape[2] <= min(shape[0], shape[1]):
        return "HWB"
    if shape[0] <= min(shape[1], shape[2]):
        return "BHW"
    return "HWB"


class NpyImporter(Importer):
    device = "npy"

    def discover(self, files, claimed):
        items = []
        for f in files:
            if f in claimed or not f.lower().endswith(".npy"):
                continue
            side, meta = _sidecar(f)
            item = ImportItem(key=f, name=stem_of(f), device=self.device, files=[f] + ([side] if side else []),
                              main_file=f)
            try:
                arr = np.load(f, mmap_mode="r")
                if arr.ndim not in (2, 3):
                    raise ImportError_(f"o cubo precisa ter 2 ou 3 dimensões (tem {arr.ndim})")
                shape = arr.shape if arr.ndim == 3 else (*arr.shape, 1)
                if _layout(shape, meta) == "BHW":
                    shape = (shape[1], shape[2], shape[0])
                item.info = dims_text(shape[0], shape[1], shape[2], meta.get("wavelengths")) + f" · {arr.dtype}"
            except (ImportError_, ValueError, OSError) as e:
                item.error = str(e)
            claimed.update(item.files)
            items.append(item)
        return items

    def load(self, item, dest_dir, link):
        side, meta = _sidecar(item.main_file)
        arr = np.load(item.main_file, mmap_mode="r")
        shape = arr.shape if arr.ndim == 3 else (*arr.shape, 1)
        layout = _layout(shape, meta)
        h, w, b = (shape[1], shape[2], shape[0]) if layout == "BHW" else shape
        dtype = arr.dtype
        del arr
        dest = place_file(item.main_file, dest_dir, link, move=item.options.get("move", False))
        if side and not link:
            place_file(side, dest_dir, link=False, move=item.options.get("move", False))

        if "value_scale" in meta:
            scale = float(meta["value_scale"])
        elif np.issubdtype(dtype, np.integer):
            scale = 1.0 / float(np.iinfo(dtype).max)
        else:
            scale = 1.0
        wavelengths = [float(x) for x in meta.get("wavelengths") or []]
        if len(wavelengths) != b:
            wavelengths = []
        names = [str(x) for x in meta.get("band_names") or []]
        if len(names) != b:
            names = [f"{x:g} nm" for x in wavelengths] or [f"banda {i + 1}" for i in range(b)]
        unit = meta.get("unit") or ("reflectance" if np.issubdtype(dtype, np.floating) else "intensity")
        if unit not in ("reflectance", "intensity"):
            raise ImportError_(f"unit inválida no .json: {unit!r}")
        storage = {"format": "npy", "path": dest if link else os.path.relpath(dest, dest_dir), "layout": layout}
        return LoadedCube(storage=storage, height=h, width=w, n_bands=b, wavelengths=wavelengths,
                          band_names=names, value_scale=scale, unit=unit,
                          mm_per_pixel=float(meta["mm_per_pixel"]) if meta.get("mm_per_pixel") else None,
                          meta={k: v for k, v in meta.items() if k not in ("wavelengths", "band_names")},
                          linked=link)
