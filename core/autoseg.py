"""Anotacao automatica: Otsu, boxes -> poligonos, SAM (Spectral Angle Mapper)
sobre o cubo inteiro, e modelos YOLO (deteccao/segmentacao/classificacao).

Portado do SpectralCrops. A diferenca: la o espectro de cada pixel eram as 8
bandas dos LEDs; aqui e o que a amostra tiver (19 do Videometer, 300 do
Resonon, ...). Com muitas bandas o SAM usa um subconjunto espacado (ver
SAM_MAX_BANDS) -- bandas vizinhas de um hiperespectral sao quase identicas,
entao o angulo praticamente nao muda e a memoria cai ~8x.
"""
import cv2
import numpy as np

from . import geometry, spectral

SAM_MIN_BRIGHTNESS = 0.7     # norma do espectro do pixel >= 70% da norma da referencia
SAM_SPLIT_AREA = 1.6         # mancha > 1,6x a area da referencia: tenta separar amostras encostadas
SAM_AREA_RANGE = (0.3, 2.5)  # area aceita, em multiplos da area media da referencia
SAM_MIN_SOLIDITY = 0.75      # area / area do envoltorio convexo
SAM_MAX_BANDS = 40
SAM_REF_PERCENTILE = 90      # limiar minimo = este percentil do angulo dos pixels da propria referencia


class AutoSegError(Exception):
    pass


# ── Otsu ──────────────────────────────────────────────────────────────────────

def otsu_polygons(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # o objeto e a regiao minoritaria (fundo ocupa a maior parte da bandeja)
    if (thresh > 0).mean() > 0.5:
        thresh = cv2.bitwise_not(thresh)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]
    valid = [c for c in contours if img_area * 0.0005 <= cv2.contourArea(c) <= img_area * 0.08]
    out = []
    for c in geometry.sort_rows(valid):
        pts = geometry.contour_to_points(c)
        if pts:
            out.append(pts)
    return out


def _otsu_contour_from_crop(crop_gray, min_area):
    blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    _, t_norm = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_inv = cv2.bitwise_not(t_norm)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def best(t):
        cs, _ = cv2.findContours(cv2.morphologyEx(t, cv2.MORPH_CLOSE, k), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        return c if cv2.contourArea(c) >= min_area else None

    cands = [c for c in (best(t_norm), best(t_inv)) if c is not None]
    return max(cands, key=cv2.contourArea) if cands else None


def polygon_from_box(img_bgr, x1, y1, x2, y2):
    h, w = img_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    gray = cv2.cvtColor(img_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
    c = _otsu_contour_from_crop(gray, (cx2 - cx1) * (cy2 - cy1) * 0.15)
    if c is None:
        return None
    pts = geometry.contour_to_points(c)
    if not pts or len(pts) < 8:
        return None
    return (np.array(pts) + [cx1, cy1]).astype(int).tolist()


# ── SAM ───────────────────────────────────────────────────────────────────────

def sam_band_subset(n_bands):
    if n_bands <= SAM_MAX_BANDS:
        return list(range(n_bands))
    return sorted(set(np.linspace(0, n_bands - 1, SAM_MAX_BANDS).round().astype(int).tolist()))


def _otsu_on_values(values):
    vmax = float(np.percentile(values, 99)) + 1e-9
    u8 = np.clip(values / vmax * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t / 255.0 * vmax


def _split_touching(contour, ref_area):
    min_dist = max(3, int(ref_area ** 0.5 * 0.45))
    x, y, bw, bh = cv2.boundingRect(contour)
    comp = np.zeros((bh + 2, bw + 2), np.uint8)
    cv2.drawContours(comp, [contour - [x - 1, y - 1]], -1, 255, -1)
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * min_dist + 1, 2 * min_dist + 1))
    cand = np.argwhere((dist == cv2.dilate(dist, k)) & (comp > 0) & (dist > 0.3 * min_dist))
    peaks = []
    for py, px in sorted(cand.tolist(), key=lambda p: -dist[p[0], p[1]]):
        if all((px - qx) ** 2 + (py - qy) ** 2 >= min_dist ** 2 for qx, qy in peaks):
            peaks.append((px, py))
    if len(peaks) < 2:
        return [contour]
    markers = np.zeros(comp.shape, np.int32)
    markers[comp == 0] = 1
    for i, (px, py) in enumerate(peaks):
        cv2.circle(markers, (px, py), 2, i + 2, -1)
    relief = (255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    markers = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
    parts = []
    for i in range(len(peaks)):
        reg = ((markers == i + 2) & (comp > 0)).astype(np.uint8) * 255
        reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            parts.append(max(cs, key=cv2.contourArea) + [x - 1, y - 1])
    return parts


def sam_polygons(sample, refs):
    """Acha as amostras com a mesma assinatura espectral dos contornos de
    referencia (poligonos/elipses ja desenhados). Devolve so os NOVOS."""
    refs = [p for p in refs if len(p) >= 3]
    if not refs:
        raise AutoSegError("Desenhe pelo menos um polígono ou elipse fechado em volta de uma amostra "
                           "pra servir de referência.")
    cube = spectral.read_bands(sample, sam_band_subset(sample.n_bands))
    cube = np.nan_to_num(cube, nan=0.0)
    h, w = cube.shape[:2]
    ref_polys = [np.array(p, dtype=np.int32) for p in refs]
    ref_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ref_mask, ref_polys, 255)
    ref_pixels = cube[ref_mask > 0]
    if len(ref_pixels) < 10:
        raise AutoSegError("A referência ficou pequena demais (ou fora da imagem).")
    ref = ref_pixels.mean(axis=0)
    ref_area = float(np.mean([abs(cv2.contourArea(p)) for p in ref_polys]))
    ref_norm = float(np.linalg.norm(ref)) + 1e-12
    px_norm = np.linalg.norm(cube, axis=2)
    cosang = (cube @ ref.astype(np.float32)) / (px_norm * ref_norm + 1e-12)
    del cube
    angles = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    bright = px_norm > SAM_MIN_BRIGHTNESS * ref_norm
    if bright.sum() < 100:
        return [], None
    # Otsu separa "parecido" de "diferente" entre os pixels claros. Se quase so
    # o alvo e claro (fundo bem escuro), o Otsu cortaria o proprio alvo ao meio:
    # o limiar nunca fica abaixo da variacao angular DENTRO da referencia.
    ref_spread = float(np.percentile(angles[ref_mask > 0], SAM_REF_PERCENTILE))
    threshold = max(_otsu_on_values(angles[bright]), ref_spread)

    mask = (bright & (angles < threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    blobs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lo, hi = SAM_AREA_RANGE[0] * ref_area, SAM_AREA_RANGE[1] * ref_area
    found = []
    for blob in blobs:
        area = cv2.contourArea(blob)
        if area < lo:
            continue
        for c in (_split_touching(blob, ref_area) if area > SAM_SPLIT_AREA * ref_area else [blob]):
            a = cv2.contourArea(c)
            hull_a = cv2.contourArea(cv2.convexHull(c))
            if lo < a < hi and hull_a and a / hull_a >= SAM_MIN_SOLIDITY:
                found.append(c)

    def overlaps_existing(c):
        m = cv2.moments(c)
        if not m["m00"]:
            return True
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        return any(cv2.pointPolygonTest(p, (cx, cy), False) >= 0 for p in ref_polys)

    polygons = []
    for c in geometry.sort_rows([c for c in found if not overlaps_existing(c)]):
        pts = geometry.contour_to_points(c)
        if pts:
            polygons.append(pts)
    return polygons, round(float(threshold), 2)


# ── YOLO (ultralytics, opcional) ──────────────────────────────────────────────

def _yolo(path):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise AutoSegError("ultralytics não está instalado neste servidor (pip install ultralytics).")
    return YOLO(path)


def yolo_boxes(model_path, img_bgr):
    res = _yolo(model_path)(img_bgr, verbose=False)[0]
    dets = [tuple(int(v) for v in xyxy) for xyxy in res.boxes.xyxy.cpu().numpy().astype(int)]
    return [list(d) for d in geometry.sort_rows(dets, rect=lambda d: (d[0], d[1], d[2] - d[0], d[3] - d[1]))]


def yolo_polygons(model_path, img_bgr):
    res = _yolo(model_path)(img_bgr, verbose=False)[0]
    if res.masks is None:
        return []
    h, w = img_bgr.shape[:2]
    contours = []
    for m in res.masks.data.cpu().numpy():
        u8 = (m * 255).astype(np.uint8)
        if u8.shape != (h, w):
            u8 = cv2.resize(u8, (w, h))
        cs, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            contours.append(max(cs, key=cv2.contourArea))
    out = []
    for c in geometry.sort_rows(contours):
        approx = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()
        closed = geometry.closed_polygon(approx)
        if closed:
            out.append([[int(x), int(y)] for x, y in closed])
    return out


def yolo_classify_crops(model_path, img_bgr, rects):
    """[label] -- um por retangulo (x1, y1, x2, y2)."""
    model = _yolo(model_path)
    h, w = img_bgr.shape[:2]
    out = []
    for x1, y1, x2, y2 in rects:
        crop = img_bgr[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
        if crop.size == 0:
            out.append("")
            continue
        res = model(crop, verbose=False)[0]
        out.append(res.names[int(res.probs.top1)] if getattr(res, "probs", None) is not None else "")
    return out
