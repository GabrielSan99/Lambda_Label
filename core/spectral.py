"""Acesso aos cubos espectrais das amostras, independente do equipamento.

Todo formato importado vira, aqui, uma visao numpy (H, W, bandas) -- memmap
sempre que possivel, entao abrir um cubo de 1 GB nao le nada do disco ate
alguem pedir uma banda ou um recorte. Os valores crus dessa visao estao na
escala do arquivo; multiplicar por sample.value_scale da a fracao 0..1.
"""
import json
import os
import threading
from collections import OrderedDict

import cv2
import numpy as np
from django.conf import settings

PREVIEW_NAME = "preview.png"
THUMB_NAME = "thumb.jpg"
META_NAME = "meta.json"
CUBE_NAME = "cube.npy"
THUMB_MAX_SIDE = 420

# Bandas usadas pra montar a cor "natural" do preview, quando a amostra tem
# comprimento de onda: a mais perto de cada alvo, se estiver a menos de
# RGB_MAX_DISTANCE_NM dele.
RGB_TARGETS_NM = (640.0, 550.0, 460.0)
RGB_MAX_DISTANCE_NM = 60.0

# Leitura em blocos de linhas: um bloco contiguo do arquivo por vez (rapido
# em BIL/BIP/npy), sem carregar o cubo inteiro quando so algumas bandas importam.
READ_CHUNK_BYTES = 64 * 1024 * 1024

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


# ── Leitura/escrita de imagem com qualquer caminho ────────────────────────────
# cv2.imread/imwrite no Windows nao abrem caminhos com acento (ex.: a pasta
# "SEED_REPORTER_CAFÉ_graõs"): le os bytes com numpy e decodifica em memoria.

def imread(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imreadmulti(path):
    """[paginas] de um TIFF multipagina (ou [] se nao der)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return []
    ok, pages = cv2.imdecodemulti(data, cv2.IMREAD_UNCHANGED)
    return list(pages) if ok and pages else []


def imwrite(path, img, params=()):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img, list(params))
    if not ok:
        raise ValueError(f"falha ao codificar {os.path.basename(path)}")
    buf.tofile(path)


def samples_root():
    return os.path.join(settings.MEDIA_ROOT, "samples")


def sample_dir(sample):
    return os.path.join(samples_root(), sample.folder)


def sample_file(sample, name):
    return os.path.join(sample_dir(sample), name)


# ── Abertura do cubo ──────────────────────────────────────────────────────────

def envi_view(path, dtype, interleave, lines, samples, bands, offset=0):
    """memmap de um arquivo ENVI como (linhas, colunas, bandas), sem copiar."""
    shapes = {"bil": (lines, bands, samples), "bip": (lines, samples, bands), "bsq": (bands, lines, samples)}
    interleave = interleave.lower()
    if interleave not in shapes:
        raise ValueError(f"interleave ENVI desconhecido: {interleave}")
    mm = np.memmap(path, dtype=np.dtype(dtype), mode="r", offset=offset, shape=shapes[interleave])
    if interleave == "bil":
        return mm.transpose(0, 2, 1)
    if interleave == "bsq":
        return mm.transpose(1, 2, 0)
    return mm


def npy_view(path, layout="HWB"):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if layout == "BHW":
        arr = arr.transpose(1, 2, 0)
    return arr


def _resolve(sample, path):
    return path if os.path.isabs(path) else os.path.join(sample_dir(sample), path)


_cube_cache = OrderedDict()
_cube_lock = threading.Lock()
CUBE_CACHE_SIZE = 4


def open_cube(sample):
    """Visao (H, W, B) do cubo cru da amostra (sem aplicar value_scale)."""
    st = sample.storage or {}
    key = (sample.id, json.dumps(st, sort_keys=True))
    with _cube_lock:
        if key in _cube_cache:
            _cube_cache.move_to_end(key)
            return _cube_cache[key]
    fmt = st.get("format")
    path = _resolve(sample, st.get("path", ""))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo do cubo não encontrado: {path}")
    if fmt == "envi":
        arr = envi_view(path, st["dtype"], st["interleave"], st["lines"], st["samples"], st["bands"],
                        st.get("offset", 0))
    elif fmt == "npy":
        arr = npy_view(path, st.get("layout", "HWB"))
    else:
        raise ValueError(f"Formato de armazenamento desconhecido: {fmt!r}")
    with _cube_lock:
        _cube_cache[key] = arr
        while len(_cube_cache) > CUBE_CACHE_SIZE:
            _cube_cache.popitem(last=False)
    return arr


def forget_cube(sample_id):
    """Solta os memmaps de uma amostra (antes de apagar os arquivos -- no
    Windows um arquivo mapeado nao pode ser removido)."""
    with _cube_lock:
        for key in [k for k in _cube_cache if k[0] == sample_id]:
            _cube_cache.pop(key, None)


def band(sample, i):
    """Banda i como float32 (H, W) ja na escala 0..1."""
    return np.asarray(open_cube(sample)[:, :, i], dtype=np.float32) * np.float32(sample.value_scale)


def read_bands(sample, idxs=None):
    """(H, W, k) float32 0..1 com as bandas pedidas (todas se idxs=None),
    lendo o arquivo em blocos de linhas contiguos."""
    arr = open_cube(sample)
    h, w, b = arr.shape
    idxs = list(range(b)) if idxs is None else list(idxs)
    out = np.empty((h, w, len(idxs)), dtype=np.float32)
    if (sample.storage or {}).get("interleave") == "bsq":
        for k, i in enumerate(idxs):        # BSQ: cada banda ja e um bloco contiguo
            out[:, :, k] = arr[:, :, i]
    else:
        row_bytes = max(1, w * b * arr.dtype.itemsize)
        step = max(1, READ_CHUNK_BYTES // row_bytes)
        for y in range(0, h, step):
            block = np.asarray(arr[y:y + step])
            out[y:y + step] = block[:, :, idxs]
    out *= np.float32(sample.value_scale)
    return out


def read_crop(sample, x0, y0, x1, y1):
    """Recorte (y0:y1, x0:x1, todas as bandas) float32 0..1."""
    arr = open_cube(sample)
    return np.asarray(arr[y0:y1, x0:x1, :], dtype=np.float32) * np.float32(sample.value_scale)


# ── Visualizacao ──────────────────────────────────────────────────────────────

def default_rgb_bands(wavelengths, n_bands):
    """[r, g, b] -- cor natural quando ha comprimentos de onda no visivel,
    senao primeira/meio/ultima banda (ou a mesma banda 3x se so ha uma)."""
    if n_bands >= 3 and len(wavelengths) == n_bands and all(w is not None for w in wavelengths):
        wl = np.array(wavelengths, dtype=float)
        picks = []
        for target in RGB_TARGETS_NM:
            i = int(np.argmin(np.abs(wl - target)))
            if abs(wl[i] - target) > RGB_MAX_DISTANCE_NM:
                break
            picks.append(i)
        if len(picks) == 3 and len(set(picks)) == 3:
            return picks
    if n_bands >= 3:
        return [n_bands - 1, n_bands // 2, 0] if n_bands > 3 else [0, 1, 2]
    return [0, 0, 0]


def stretch_u8(img, lo=1.0, hi=99.0, mask=None):
    """Contraste por percentil (por canal) -> uint8. NaN vira 0."""
    img = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0)
    if img.ndim == 2:
        img = img[:, :, None]
    out = np.empty(img.shape, dtype=np.uint8)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        sample_vals = ch[mask] if mask is not None else ch[::4, ::4]
        a, b = np.percentile(sample_vals, [lo, hi]) if sample_vals.size else (0.0, 1.0)
        if b - a < 1e-9:
            b = a + 1e-9
        out[:, :, c] = np.clip((ch - a) / (b - a) * 255.0, 0, 255).astype(np.uint8)
    return out[:, :, 0] if out.shape[2] == 1 else out


def composite_bgr(sample, rgb_bands=None):
    """Composicao colorida (BGR uint8, pronta pro cv2) a partir de 3 bandas."""
    r, g, b = rgb_bands or sample.rgb_bands or default_rgb_bands(sample.wavelengths, sample.n_bands)
    stack = read_bands(sample, [b, g, r])
    return stretch_u8(stack)


_SPECTRAL_LUT = None


def spectral_lut():
    """LUT BGR do colormap nipy_spectral (o mesmo das bandas do SpectralCrops),
    gerada sem depender do matplotlib."""
    global _SPECTRAL_LUT
    if _SPECTRAL_LUT is None:
        # pontos de controle do nipy_spectral (posicao, r, g, b)
        stops = np.array([
            [0.00, 0.0000, 0.0000, 0.0000], [0.05, 0.4667, 0.0000, 0.5333], [0.10, 0.5333, 0.0000, 0.6000],
            [0.15, 0.0000, 0.0000, 0.6667], [0.20, 0.0000, 0.0000, 0.8667], [0.25, 0.0000, 0.4667, 0.8667],
            [0.30, 0.0000, 0.6000, 0.8667], [0.35, 0.0000, 0.6667, 0.6667], [0.40, 0.0000, 0.6667, 0.5333],
            [0.45, 0.0000, 0.6000, 0.0000], [0.50, 0.0000, 0.7333, 0.0000], [0.55, 0.0000, 0.8667, 0.0000],
            [0.60, 0.0000, 1.0000, 0.0000], [0.65, 0.7333, 1.0000, 0.0000], [0.70, 0.9333, 0.9333, 0.0000],
            [0.75, 1.0000, 0.8000, 0.0000], [0.80, 1.0000, 0.6000, 0.0000], [0.85, 1.0000, 0.0000, 0.0000],
            [0.90, 0.8667, 0.0000, 0.0000], [0.95, 0.8000, 0.0000, 0.0000], [1.00, 0.8000, 0.8000, 0.8000],
        ])
        x = np.linspace(0, 1, 256)
        rgb = np.stack([np.interp(x, stops[:, 0], stops[:, k]) for k in (1, 2, 3)], axis=1)
        _SPECTRAL_LUT = (rgb[:, ::-1] * 255).astype(np.uint8).reshape(256, 1, 3)
    return _SPECTRAL_LUT


def render_band(sample, i, cmap="spectral", fixed_range=True):
    """PNG (BGR uint8) de uma banda. fixed_range: 0..100% de reflectancia
    vira 0..255 (a mesma cor = a mesma reflectancia em qualquer banda, como
    nas miniaturas do SpectralCrops); senao contraste por percentil."""
    data = band(sample, i)
    if fixed_range and sample.unit == sample.Unit.REFLECTANCE:
        gray = np.clip(np.nan_to_num(data, nan=0.0) * 255.0, 0, 255).astype(np.uint8)
    else:
        gray = stretch_u8(data)
    if cmap == "spectral":
        return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), spectral_lut())
    return gray


def write_previews(sample, bgr=None):
    """Grava preview.png (fundo do editor) e thumb.jpg (grade)."""
    if bgr is None:
        bgr = composite_bgr(sample)
    os.makedirs(sample_dir(sample), exist_ok=True)
    imwrite(sample_file(sample, PREVIEW_NAME), bgr)
    h, w = bgr.shape[:2]
    s = min(1.0, THUMB_MAX_SIDE / max(h, w))
    thumb = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    imwrite(sample_file(sample, THUMB_NAME), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_meta(sample):
    """meta.json na pasta da amostra -- deixa a pasta autoexplicativa mesmo fora do app."""
    meta = {
        "name": sample.name, "device": sample.device, "source_path": sample.source_path,
        "linked": sample.linked, "storage": sample.storage,
        "height": sample.height, "width": sample.width, "n_bands": sample.n_bands,
        "wavelengths": sample.wavelengths, "band_names": sample.band_names,
        "value_scale": sample.value_scale, "unit": sample.unit,
        "mm_per_pixel": sample.mm_per_pixel, "rgb_bands": sample.rgb_bands,
        "header": sample.meta,
    }
    with open(sample_file(sample, META_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def load_preview(sample):
    """Imagem de referencia (BGR) usada por Otsu/YOLO -- a mesma que o usuario ve."""
    return imread(sample_file(sample, PREVIEW_NAME), cv2.IMREAD_COLOR)


# ── Espectro de ROIs ──────────────────────────────────────────────────────────

POINT_RADIUS_PX = 3   # janela (2r+1)^2 ao redor de um ROI tipo ponto


def contour_bbox(contour, h, w):
    """(x0, y0, x1, y1) inteiro, recortado a imagem, da regiao do ROI."""
    kind = contour.get("shape")
    if kind == "box":
        xs = sorted((contour["x1"], contour["x2"]))
        ys = sorted((contour["y1"], contour["y2"]))
    elif kind == "point":
        r = POINT_RADIUS_PX
        xs = [contour["x"] - r, contour["x"] + r + 1]
        ys = [contour["y"] - r, contour["y"] + r + 1]
    else:
        pts = contour.get("points") or []
        if len(pts) < 3:
            return None
        xs = [min(p[0] for p in pts), max(p[0] for p in pts) + 1]
        ys = [min(p[1] for p in pts), max(p[1] for p in pts) + 1]
    x0, y0 = max(0, int(np.floor(xs[0]))), max(0, int(np.floor(ys[0])))
    x1, y1 = min(w, int(np.ceil(xs[1]))), min(h, int(np.ceil(ys[1])))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def contour_mask(contour, bbox):
    """Mascara bool do ROI dentro do recorte bbox."""
    x0, y0, x1, y1 = bbox
    kind = contour.get("shape")
    if kind in ("box", "point"):
        return np.ones((y1 - y0, x1 - x0), dtype=bool)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    pts = np.array(contour["points"], dtype=np.float64) - [x0, y0]
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def roi_pixels(sample, contour):
    """(N, B) float32 0..1 dos pixels dentro do ROI, ou None se vazio."""
    bbox = contour_bbox(contour, sample.height, sample.width)
    if bbox is None:
        return None
    mask = contour_mask(contour, bbox)
    if not mask.any():
        return None
    crop = read_crop(sample, *bbox)
    return crop[mask]


def roi_spectrum(sample, contour):
    """(media %, desvio %, n_pixels) por banda -- None nas bandas sem pixel finito."""
    px = roi_pixels(sample, contour)
    if px is None:
        return None, None, 0
    finite = np.isfinite(px)
    n = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(px, axis=0) / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, np.nansum((px - mean) ** 2, axis=0) / np.maximum(n - 1, 1), np.nan)
    to_list = lambda a: [None if not np.isfinite(v) else round(float(v) * 100, 4) for v in a]
    return to_list(mean), to_list(np.sqrt(var)), int(px.shape[0])


def is_monotonic(wavelengths):
    return (len(wavelengths) >= 2 and all(w is not None for w in wavelengths)
            and all(b > a for a, b in zip(wavelengths, wavelengths[1:])))
