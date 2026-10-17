"""VideometerLab (.hips) -- reflectancia (%) em ~19 bandas de LED, ou
reflectancia + fluorescencia. Lido pelo leitor Python puro do pacote oficial
`videometer` (videometer.hips_core, sem as DLLs .NET), e convertido uma vez
pra cube.npy: o .hips pode ter bandas comprimidas, entao nao da memmap direto.
"""
import os

import numpy as np

from .. import spectral
from .base import ImportError_, ImportItem, Importer, LoadedCube, dims_text, place_file, stem_of


def _hips():
    try:
        from videometer.hips_core import HipsImage
    except ImportError as e:   # pragma: no cover - depende do ambiente
        raise ImportError_("Pacote 'videometer' não instalado (pip install videometer==2.0.4).") from e
    return HipsImage


class VideometerImporter(Importer):
    device = "videometer"

    def discover(self, files, claimed):
        items = []
        for f in files:
            if f in claimed or not f.lower().endswith(".hips"):
                continue
            item = ImportItem(key=f, name=stem_of(f), device=self.device, files=[f], main_file=f)
            try:
                hdr = _hips().read_header(f)
                item.info = dims_text(hdr.height, hdr.width, hdr.bands, [float(x) for x in hdr.wavelengths])
                if hdr.mm_pixel:
                    item.info += f" · {hdr.mm_pixel * 1000:.1f} µm/px"
            except (ImportError_, ValueError, OSError) as e:
                item.error = str(e)
            claimed.add(f)
            items.append(item)
        return items

    def load(self, item, dest_dir, link):
        HipsImage = _hips()
        try:
            img = HipsImage.read(item.main_file)
            pixels = np.asarray(img.pixels, dtype=np.float32)
        except Exception as e:
            raise ImportError_(f"Falha ao ler o .hips: {e}") from e
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        h, w, b = pixels.shape
        np.save(os.path.join(dest_dir, spectral.CUBE_NAME), pixels)
        del pixels
        if not link:
            # guarda o original junto (o .hips tem camadas/historico que o cube.npy nao tem)
            place_file(item.main_file, dest_dir, link=False, move=item.options.get("move", False))

        wavelengths = [float(x) for x in img.wavelengths] if len(img.wavelengths) == b else []
        names = [str(n) for n in img.band_names] if len(img.band_names) == b else []
        if not names:
            names = [f"{x:g} nm" for x in wavelengths] or [f"banda {i + 1}" for i in range(b)]
        meta = {"description": img.description, "illumination": list(img.illumination_names),
                "strobe_times": [float(x) for x in img.strobe_times],
                "extra_data": {k: float(v) for k, v in img.extra_data.items()},
                "extra_data_string": dict(img.extra_data_string)}
        # O Videometer grava reflectancia em %: 0..100 -> fracao 0..1.
        return LoadedCube(storage={"format": "npy", "path": spectral.CUBE_NAME, "layout": "HWB"},
                          height=h, width=w, n_bands=b, wavelengths=wavelengths, band_names=names,
                          value_scale=0.01, unit="reflectance",
                          mm_per_pixel=float(img.mm_pixel) if img.mm_pixel else None, meta=meta)
