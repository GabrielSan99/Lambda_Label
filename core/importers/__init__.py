"""Importacao de imagens ja coletadas por outros equipamentos.

Fluxo: scan(caminhos) -> [ImportItem] (o usuario confere e escolhe) ->
import_item(projeto, item) -> Sample pronta pra anotar.

A ordem de IMPORTERS importa: cada um reivindica os arquivos que reconhece,
e o de imagens comuns fica por ultimo pra pegar so o que sobrou.
"""
import os
import shutil
import uuid

from .. import spectral
from .base import ImportError_, ImportItem  # noqa: F401  (reexport)
from .envi import EnviImporter
from .images import ImageImporter
from .npy import NpyImporter
from .spectralcrops import SpectralCropsImporter
from .videometer import VideometerImporter

MAX_SCAN_FILES = 50000
SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", "node_modules", "_staging"}


def build_importers(group_by_prefix=True):
    return [EnviImporter(), VideometerImporter(), NpyImporter(), SpectralCropsImporter(),
            ImageImporter(group_by_prefix=group_by_prefix)]


def importer_for(device):
    for imp in build_importers():
        if imp.device == device:
            return imp
    raise ImportError_(f"Formato desconhecido: {device}")


def list_files(roots):
    out = []
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            out.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                out.append(os.path.join(dirpath, fn))
                if len(out) > MAX_SCAN_FILES:
                    raise ImportError_(f"Mais de {MAX_SCAN_FILES} arquivos — escolha uma pasta mais específica.")
    return out


def scan(roots, group_by_prefix=True):
    """(itens encontrados, arquivos ignorados)."""
    files = list_files(roots)
    claimed, items = set(), []
    for imp in build_importers(group_by_prefix):
        items.extend(imp.discover(files, claimed))
    ignored = [f for f in files if f not in claimed]
    return items, ignored


def _unique_name(project, name):
    existing = set(project.samples.values_list("name", flat=True))
    if name not in existing:
        return name
    n = 2
    while f"{name} ({n})" in existing:
        n += 1
    return f"{name} ({n})"


def import_item(project, item, link=False, move=False):
    """Cria a Sample (status pending -> ready/error). Nunca levanta: o erro
    fica gravado na propria amostra pra aparecer na grade."""
    from ..models import Annotation, Sample

    sample = Sample.objects.create(project=project, name=_unique_name(project, item.name), device=item.device,
                                   source_path=item.main_file, folder=f"{project.id}/{uuid.uuid4().hex}")
    dest = spectral.sample_dir(sample)
    os.makedirs(dest, exist_ok=True)
    item.options["move"] = move and not link
    try:
        loaded = importer_for(item.device).load(item, dest, link)
        for field in ("storage", "height", "width", "n_bands", "wavelengths", "band_names", "value_scale",
                      "unit", "mm_per_pixel", "meta", "linked"):
            setattr(sample, field, getattr(loaded, field))
        sample.rgb_bands = loaded.rgb_bands or spectral.default_rgb_bands(loaded.wavelengths, loaded.n_bands)
        spectral.write_previews(sample, loaded.preview_bgr)
        spectral.write_meta(sample)
        sample.status = Sample.Status.READY
        sample.error = ""
        sample.save()
        Annotation.objects.get_or_create(sample=sample)
    except Exception as e:  # erro de leitura de qualquer tipo vira mensagem na amostra
        spectral.forget_cube(sample.id)
        sample.status = Sample.Status.ERROR
        sample.error = str(e) or e.__class__.__name__
        sample.save()
        for name in os.listdir(dest):   # nao deixa cubo convertido pela metade ocupando disco
            path = os.path.join(dest, name)
            (shutil.rmtree if os.path.isdir(path) else os.remove)(path)
    return sample
