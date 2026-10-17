"""Base dos importadores: cada formato descobre os seus arquivos dentro do que
o usuario mandou (upload ou pasta local) e sabe converter um item numa Sample.

Pra suportar um equipamento novo: crie um modulo com uma subclasse de
Importer (discover + load) e registre em importers/__init__.py:IMPORTERS.
"""
import os
import shutil
from dataclasses import dataclass, field


@dataclass
class ImportItem:
    """Uma amostra encontrada na varredura, ainda nao importada."""
    key: str                      # id unico dentro da varredura
    name: str
    device: str
    files: list                   # caminhos absolutos que pertencem a essa amostra
    main_file: str                # arquivo que identifica a amostra (cabecalho, .hips, ...)
    info: str = ""                # "2192 x 2192 px · 19 bandas · 365–970 nm"
    error: str = ""
    options: dict = field(default_factory=dict)

    @property
    def size(self):
        return sum(os.path.getsize(f) for f in self.files if os.path.exists(f))

    def to_json(self):
        return {"key": self.key, "name": self.name, "device": self.device, "info": self.info,
                "error": self.error, "size": self.size, "n_files": len(self.files),
                "main_file": os.path.basename(self.main_file)}


@dataclass
class LoadedCube:
    """O que um importador devolve: como abrir o cubo + metadados da amostra."""
    storage: dict
    height: int
    width: int
    n_bands: int
    wavelengths: list
    band_names: list
    value_scale: float
    unit: str
    mm_per_pixel: float = None
    rgb_bands: list = None
    meta: dict = field(default_factory=dict)
    preview_bgr: object = None    # preview pronto (ex.: rgb.png do SpectralCrops); None = gera da composicao
    linked: bool = False


class ImportError_(Exception):
    """Erro de importacao com mensagem pronta pro usuario."""


class Importer:
    device = ""

    def discover(self, files, claimed):
        """Recebe todos os arquivos encontrados (caminhos absolutos) e o set dos
        ja reivindicados por outro importador; devolve [ImportItem] e adiciona
        a `claimed` os arquivos que usou."""
        raise NotImplementedError

    def load(self, item, dest_dir, link):
        """Le o item e devolve LoadedCube. dest_dir = pasta da amostra (ja
        existe). link=True: pode ler direto do arquivo original, sem copiar
        (so faz diferenca pra formatos que dao memmap direto, como ENVI)."""
        raise NotImplementedError


def place_file(src, dest_dir, link, move=False):
    """Copia/move `src` pra dest_dir/source/ (ou so devolve src se link)."""
    if link:
        return os.path.abspath(src)
    target_dir = os.path.join(dest_dir, "source")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, os.path.basename(src))
    if move:
        shutil.move(src, target)
    else:
        shutil.copy2(src, target)
    return target


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def dims_text(h, w, b, wavelengths=None):
    txt = f"{w} × {h} px · {b} banda{'s' if b != 1 else ''}"
    wl = [x for x in (wavelengths or []) if x is not None]
    if wl:
        txt += f" · {min(wl):g}–{max(wl):g} nm"
    return txt
