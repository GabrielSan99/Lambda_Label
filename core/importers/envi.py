"""Cubos hiperespectrais com cabecalho ENVI (.hdr) -- Resonon/Spectronon,
Specim, Headwall, HySpex e a maioria das cameras push-broom exportam assim.

O arquivo de dados e lido via memmap (spectral.envi_view): no modo
"vincular" nada e copiado, o que importa com cubos de ~1 GB cada.
"""
import os
import re

import numpy as np

from .. import spectral
from .base import ImportError_, ImportItem, Importer, LoadedCube, dims_text, place_file

DATA_EXTS = ("", ".bil", ".bip", ".bsq", ".raw", ".img", ".dat", ".cube")


def parse_header(path):
    """Cabecalho ENVI -> dict (chaves em minusculas). Listas `{a, b}` viram
    listas de strings; o resto fica string."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if not text.lstrip().upper().startswith("ENVI"):
        raise ImportError_(f"{os.path.basename(path)} não é um cabeçalho ENVI.")
    out = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]*)", text, flags=re.M | re.S):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith("{"):
            inner = val[1:-1].strip()
            out[key] = [v.strip() for v in inner.split(",")] if inner else []
        else:
            out[key] = val
    return out


def _float_list(vals):
    out = []
    for v in vals or []:
        try:
            out.append(float(v))
        except ValueError:
            return []
    return out


def find_data_file(hdr_path, available):
    """O arquivo de dados que acompanha o .hdr: 'X.bil.hdr' -> 'X.bil';
    'X.hdr' -> 'X', 'X.bil', 'X.raw', ..."""
    base = hdr_path[:-4]
    candidates = [base] if os.path.splitext(base)[1] else [base + ext for ext in DATA_EXTS]
    lower = {p.lower(): p for p in available}
    for c in candidates:
        if c.lower() in lower and os.path.isfile(lower[c.lower()]):
            return lower[c.lower()]
    return None


class EnviImporter(Importer):
    device = "envi"

    def discover(self, files, claimed):
        items = []
        for f in files:
            if f in claimed or not f.lower().endswith(".hdr"):
                continue
            data = find_data_file(f, files)
            name = os.path.basename(f)[:-4]
            name = os.path.splitext(name)[0] if os.path.splitext(name)[1].lower() in DATA_EXTS else name
            item = ImportItem(key=f, name=name, device=self.device, files=[f] + ([data] if data else []),
                              main_file=f)
            try:
                hdr = parse_header(f)
                h, w, b = int(hdr["lines"]), int(hdr["samples"]), int(hdr["bands"])
                item.info = dims_text(h, w, b, _float_list(hdr.get("wavelength"))) + f" · {hdr.get('interleave', '?').upper()}"
                if not data:
                    item.error = "Arquivo de dados (.bil/.bip/.bsq/.raw) não encontrado ao lado do .hdr."
            except (ImportError_, KeyError, ValueError) as e:
                item.error = f"Cabeçalho inválido: {e}"
            claimed.update(item.files)
            items.append(item)
        return items

    def load(self, item, dest_dir, link):
        hdr_path = item.main_file
        hdr = parse_header(hdr_path)
        try:
            h, w, b = int(hdr["lines"]), int(hdr["samples"]), int(hdr["bands"])
            dtype_code = int(hdr.get("data type", 4))
        except (KeyError, ValueError) as e:
            raise ImportError_(f"Cabeçalho ENVI incompleto ({e}).")
        if dtype_code not in spectral.ENVI_DTYPES:
            raise ImportError_(f"Tipo de dado ENVI {dtype_code} não suportado.")
        dtype = (">" if hdr.get("byte order", "0").strip() == "1" else "<") + spectral.ENVI_DTYPES[dtype_code]
        interleave = hdr.get("interleave", "bsq").strip().lower()
        offset = int(hdr.get("header offset", 0) or 0)
        data_files = [f for f in item.files if f != hdr_path]
        if not data_files:
            raise ImportError_("Arquivo de dados do cubo ENVI não encontrado.")
        data = data_files[0]
        expected = offset + h * w * b * np.dtype(dtype).itemsize
        actual = os.path.getsize(data)
        if actual < expected:
            raise ImportError_(f"Arquivo de dados menor que o esperado pelo cabeçalho "
                               f"({actual:,} de {expected:,} bytes) — cópia incompleta?")

        hdr_dest = place_file(hdr_path, dest_dir, link, move=item.options.get("move", False))
        data_dest = place_file(data, dest_dir, link, move=item.options.get("move", False))

        wavelengths = _float_list(hdr.get("wavelength"))
        if hdr.get("wavelength units", "nm").lower().startswith("micro") or (wavelengths and max(wavelengths) < 30):
            wavelengths = [x * 1000 for x in wavelengths]       # um -> nm
        if len(wavelengths) != b:
            wavelengths = []
        names = hdr.get("band names") or []
        if len(names) != b:
            names = [f"{x:g} nm" for x in wavelengths] if wavelengths else [f"banda {i + 1}" for i in range(b)]

        # reflectancia: "reflectance scale factor" (Resonon grava 1.0 -> ja e fracao;
        # muitos softwares gravam 10000 em inteiros). Sem a chave: float = fracao,
        # inteiro = intensidade relativa ao fundo de escala do tipo.
        rsf = hdr.get("reflectance scale factor")
        if rsf:
            scale, unit = 1.0 / float(rsf), "reflectance"
        elif dtype_code in (4, 5):
            scale, unit = 1.0, "reflectance"
        else:
            scale, unit = 1.0 / float(np.iinfo(np.dtype(dtype)).max), "intensity"

        rgb = None
        if hdr.get("default bands"):
            try:
                rgb = [int(v) - 1 for v in hdr["default bands"]][:3]
                rgb = rgb if len(rgb) == 3 and all(0 <= v < b for v in rgb) else None
            except ValueError:
                rgb = None

        storage = {"format": "envi", "path": data_dest if link else os.path.relpath(data_dest, dest_dir),
                   "header": hdr_dest if link else os.path.relpath(hdr_dest, dest_dir),
                   "dtype": dtype, "interleave": interleave, "lines": h, "samples": w, "bands": b,
                   "offset": offset}
        keep = {k: v for k, v in hdr.items() if k not in ("wavelength", "band names", "fwhm")}
        return LoadedCube(storage=storage, height=h, width=w, n_bands=b, wavelengths=wavelengths,
                          band_names=names, value_scale=scale, unit=unit, rgb_bands=rgb, meta=keep,
                          linked=link)
