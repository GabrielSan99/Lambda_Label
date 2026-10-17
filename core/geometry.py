"""Geometria das anotacoes -- portado do SpectralCrops (views.py), sem
dependencia de equipamento: tudo em pixels da imagem da amostra."""
import cv2
import numpy as np

SHAPE_ORDER = ("polygon", "ellipse", "box", "point")   # prioridade da label da AMOSTRA


def closed_polygon(pts):
    """Vertices de um contorno fechado valido (>=3 pontos, area > 0), sem o
    ponto final repetindo o inicial -- ou None."""
    try:
        pts = [(float(x), float(y)) for x, y in pts]
    except (TypeError, ValueError):
        return None
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    area2 = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    return pts if area2 != 0 else None


def keep_closed(shapes, labels):
    """So os contornos fechados validos (e seus labels, alinhados) + quantos cairam."""
    labels = list(labels or []) + [""] * max(0, len(shapes) - len(labels or []))
    kept = [(closed_polygon(s), l) for s, l in zip(shapes, labels)]
    kept = [(s, l) for s, l in kept if s]
    return [s for s, _ in kept], [l for _, l in kept], len(shapes) - len(kept)


def area_px(contour):
    kind = contour.get("shape")
    if kind == "box":
        return abs(contour["x2"] - contour["x1"]) * abs(contour["y2"] - contour["y1"])
    if kind == "point":
        return None
    pts = contour.get("points") or []
    if len(pts) < 3:
        return None
    a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return abs(a) / 2.0


def bbox_px(contour):
    """(largura, altura) da caixa ALINHADA AOS EIXOS DA IMAGEM -- referencia
    aproximada (objeto rotacionado infla os dois valores)."""
    kind = contour.get("shape")
    if kind == "box":
        return abs(contour["x2"] - contour["x1"]), abs(contour["y2"] - contour["y1"])
    if kind == "point":
        return None, None
    pts = contour.get("points") or []
    if len(pts) < 2:
        return None, None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def annotation_rois(ann):
    """[(method, label, contour)] de todas as formas da anotacao, no formato
    de ROIMeasurement.contour."""
    rois = []
    pad = lambda lbls, n: list(lbls or []) + [""] * max(0, n - len(lbls or []))
    for box, lbl in zip(ann.boxes, pad(ann.labels, len(ann.boxes))):
        rois.append(("box", lbl, {"shape": "box", "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}))
    for pts, lbl in zip(ann.ellipses, pad(ann.ellipse_labels, len(ann.ellipses))):
        rois.append(("ellipse", lbl, {"shape": "ellipse", "points": pts}))
    for pts, lbl in zip(ann.polygons, pad(ann.poly_labels, len(ann.polygons))):
        rois.append(("polygon", lbl, {"shape": "polygon", "points": pts}))
    for pt, lbl in zip(ann.points, pad(ann.point_labels, len(ann.points))):
        rois.append(("point", lbl, {"shape": "point", "x": pt[0], "y": pt[1]}))
    return rois


def sort_rows(items, rect=cv2.boundingRect):
    """Ordena em linhas (cima -> baixo) e, na linha, da esquerda pra direita.
    `rect(item)` -> (x, y, w, h)."""
    if not items:
        return []
    heights = sorted(rect(c)[3] for c in items)
    tol = max(10, heights[len(heights) // 2] // 2)
    rows, current = [], []
    for c in sorted(items, key=lambda c: rect(c)[1]):
        if not current or abs(rect(c)[1] - rect(current[0])[1]) <= tol:
            current.append(c)
        else:
            rows.append(current)
            current = [c]
    rows.append(current)
    return [c for row in rows for c in sorted(row, key=lambda c: rect(c)[0])]


def contour_to_points(c, max_points=40):
    """Contorno do opencv -> lista de pontos simplificada (>= 8 pontos quando der)."""
    approx = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) < 8:
        raw = c.reshape(-1, 2)
        approx = raw[::max(1, len(raw) // max_points)]
    return approx.tolist() if len(approx) >= 4 else None


def group_samples(ann):
    """Agrupa os contornos que caem na mesma amostra (semente, folha...): dois
    contornos sao do mesmo grupo quando o centro de um esta dentro do outro;
    um ponto entra na menor forma que o contem. Numerado em linhas. Mesma
    saida do export de amostras do SpectralCrops (anotacoes.json)."""
    img_label = (ann.image_label or "").strip()
    shapes = []

    def add(kind, items, labels, to_contour):
        labels = list(labels or [])
        for i, geom in enumerate(items or []):
            lbl = ((labels[i] if i < len(labels) else "") or "").strip()
            c = to_contour(geom)
            if c is not None:
                shapes.append({"kind": kind, "idx": i, "geom": geom, "label": lbl, "contour": c})

    def poly_c(pts):
        pts = closed_polygon(pts)
        return np.array(pts, dtype=np.float32) if pts else None

    def box_c(b):
        x1, y1, x2, y2 = (float(v) for v in b)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    add("polygon", ann.polygons, ann.poly_labels, poly_c)
    add("ellipse", ann.ellipses, ann.ellipse_labels, poly_c)
    add("box", ann.boxes, ann.labels, box_c)
    add("point", ann.points, ann.point_labels, lambda p: np.array([[float(p[0]), float(p[1])]], dtype=np.float32))

    for s in shapes:
        c = s["contour"]
        m = cv2.moments(c) if len(c) >= 3 else None
        s["center"] = ((m["m10"] / m["m00"], m["m01"] / m["m00"]) if m and m["m00"]
                       else (float(c[:, 0].mean()), float(c[:, 1].mean())))
        s["area"] = abs(cv2.contourArea(c)) if len(c) >= 3 else 0.0

    parent = list(range(len(shapes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def inside(pt, s):
        return len(s["contour"]) >= 3 and cv2.pointPolygonTest(s["contour"], pt, False) >= 0

    areas = [i for i, s in enumerate(shapes) if s["kind"] != "point"]
    for a in areas:
        for b in areas:
            if a < b and (inside(shapes[a]["center"], shapes[b]) or inside(shapes[b]["center"], shapes[a])):
                parent[find(a)] = find(b)
    for i, s in enumerate(shapes):
        if s["kind"] == "point":
            hosts = [j for j in areas if inside(s["center"], shapes[j])]
            if hosts:
                parent[find(i)] = find(min(hosts, key=lambda j: shapes[j]["area"]))

    groups = {}
    for i in range(len(shapes)):
        groups.setdefault(find(i), []).append(shapes[i])
    groups = list(groups.values())
    if not groups:
        return []

    def grect(g):
        pts = np.concatenate([s["contour"] for s in g])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = max(g, key=lambda s: s["area"])["center"]
        # sort_rows ordena pelo canto (x, y); usa o centro da forma principal
        return (cx, cy, x1 - x0, y1 - y0)

    samples = []
    for n, g in enumerate(sort_rows(groups, rect=grect), start=1):
        g.sort(key=lambda s: (SHAPE_ORDER.index(s["kind"]), s["idx"]))
        own = next((s["label"] for s in g if s["label"]), "")
        contours = []
        for s in g:
            geom = s["geom"]
            if s["kind"] in ("polygon", "ellipse"):
                geom = [[int(round(x)), int(round(y))] for x, y in closed_polygon(geom)]
                key = "pontos"
            elif s["kind"] == "box":
                geom = [int(round(v)) for v in geom]
                key = "caixa"
            else:
                geom = [int(round(geom[0])), int(round(geom[1]))]
                key = "ponto"
            contours.append({"tipo": s["kind"], "indice": s["idx"] + 1, "label": s["label"] or img_label,
                             "label_origem": "amostra" if s["label"] else ("imagem" if img_label else None),
                             key: geom})
        samples.append({"amostra": n, "label": own or img_label,
                        "label_origem": "amostra" if own else ("imagem" if img_label else None),
                        "contornos": contours})
    return samples
