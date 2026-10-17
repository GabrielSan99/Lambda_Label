"""Classificador por assinatura espectral (scikit-learn), treinado com os
espectros medios dos ROIs (ROIMeasurement) do projeto.

Diferencas em relacao ao SpectralCrops:
- o numero de bandas nao e fixo. As features seguem a grade de bandas da
  "assinatura" mais comum do projeto; amostras de outro equipamento entram
  interpoladas nessa grade quando cobrem a mesma faixa de comprimento de onda
  (senao ficam de fora, e a tela diz quantas);
- o rotulo pode vir da amostra inteira (rotulo da imagem, a "condicao" do
  experimento) ou de cada forma (label do ROI);
- validacao cruzada por AMOSTRA (StratifiedGroupKFold) sempre que da: ROIs
  da mesma imagem compartilham luz/lote/posicao, e cair dos dois lados do
  split infla a acuracia.
"""
import io
import os
import threading
import time

import numpy as np

ALGORITHMS = {
    "random_forest": {
        "label": "Random Forest",
        "params": {
            "n_estimators":     {"type": "int", "min": 10, "max": 1000, "step": 10, "default": 200},
            "max_depth":        {"type": "int_or_none", "min": 1, "max": 50, "step": 1, "default": None},
            "min_samples_leaf": {"type": "int", "min": 1, "max": 20, "step": 1, "default": 1},
        },
    },
    "svm": {
        "label": "SVM",
        "params": {
            "C":      {"type": "float", "min": 0.01, "max": 1000.0, "step": 0.01, "default": 1.0},
            "kernel": {"type": "choice", "choices": ["linear", "rbf", "poly"], "default": "rbf"},
            "gamma":  {"type": "choice", "choices": ["scale", "auto"], "default": "scale"},
        },
    },
    "logistic_regression": {
        "label": "Regressão Logística",
        "params": {
            "C":        {"type": "float", "min": 0.001, "max": 1000.0, "step": 0.001, "default": 1.0},
            "max_iter": {"type": "int", "min": 100, "max": 5000, "step": 100, "default": 1000},
        },
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "params": {
            "n_neighbors": {"type": "int", "min": 1, "max": 50, "step": 1, "default": 5},
            "weights":     {"type": "choice", "choices": ["uniform", "distance"], "default": "uniform"},
        },
    },
}

LABEL_SOURCES = {"image": "Rótulo da imagem (condição)", "roi": "Label de cada forma (ROI)"}
WL_TOLERANCE_NM = 2.0   # folga nas pontas ao decidir se uma amostra cobre a grade de referencia


class MLError(Exception):
    pass


def clamp_params(algorithm, raw):
    """Hiperparametros do cliente -> dentro dos limites de ALGORITHMS."""
    spec = ALGORITHMS[algorithm]["params"]
    raw = raw or {}
    out = {}
    for name, p in spec.items():
        val = raw.get(name, p["default"])
        try:
            if p["type"] == "int":
                val = max(p["min"], min(p["max"], int(val)))
            elif p["type"] == "int_or_none":
                val = None if val in (None, "", "none", "None") else max(p["min"], min(p["max"], int(val)))
            elif p["type"] == "float":
                val = max(p["min"], min(p["max"], float(val)))
            elif p["type"] == "choice" and val not in p["choices"]:
                val = p["default"]
        except (TypeError, ValueError):
            val = p["default"]
        out[name] = val
    return out


def build_estimator(algorithm, params):
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"], random_state=42, n_jobs=1)
    if algorithm == "svm":
        from sklearn.svm import SVC
        return SVC(C=params["C"], kernel=params["kernel"], gamma=params["gamma"], random_state=42)
    if algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    if algorithm == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights=params["weights"])
    raise MLError(f"algoritmo desconhecido: {algorithm}")


def make_pipeline(algorithm, params):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("clf", build_estimator(algorithm, params))])


# ── Alinhamento de bandas ─────────────────────────────────────────────────────

def signature(sample):
    """Identifica o conjunto de bandas: comprimentos de onda (arredondados) ou, sem eles, so a contagem."""
    if sample.wavelengths and all(w is not None for w in sample.wavelengths):
        return ("wl",) + tuple(round(float(w), 1) for w in sample.wavelengths)
    return ("n", sample.n_bands)


def _monotonic(wl):
    return len(wl) >= 2 and all(b > a for a, b in zip(wl, wl[1:]))


def align(values, src_wl, src_sig, ref):
    """Espectro (lista na ordem da amostra) -> vetor na grade de referencia, ou None."""
    if values is None or any(v is None for v in values):
        return None
    vec = np.asarray(values, dtype=float)
    if src_sig == ref["signature"]:
        return vec
    ref_wl = ref.get("wavelengths")
    if not ref_wl or not src_wl or not _monotonic(src_wl) or not _monotonic(ref_wl):
        return None
    if src_wl[0] > ref_wl[0] + WL_TOLERANCE_NM or src_wl[-1] < ref_wl[-1] - WL_TOLERANCE_NM:
        return None
    return np.interp(ref_wl, src_wl, vec)


def roi_dataset(project, label_source="image", unlabeled_as=None):
    """X, y, grupos (id da amostra), referencia de bandas e contagens do que ficou de fora.
    unlabeled_as: rotulo dado aos ROIs sem rotulo (a Analise mostra "sem grupo");
    None = ficam de fora (treino)."""
    from .models import ROIMeasurement
    rois = list(ROIMeasurement.objects.filter(project=project)
                .select_related("sample", "sample__annotation").order_by("sample_id", "method", "index"))

    def label_of(r):
        if label_source == "roi":
            return (r.label or "").strip()
        ann = getattr(r.sample, "annotation", None)
        return ((ann.image_label if ann else "") or "").strip()

    labeled = [(r, label_of(r) or unlabeled_as) for r in rois]
    labeled = [(r, l) for r, l in labeled if l]
    info = {"n_rois": len(rois), "unlabeled": len(rois) - len(labeled), "incompatible": 0, "incomplete": 0}
    if not labeled:
        return np.empty((0, 0)), np.array([]), np.array([]), None, info

    counts = {}
    for r, _ in labeled:
        sig = signature(r.sample)
        counts[sig] = counts.get(sig, 0) + 1
    ref_sig = max(counts, key=counts.get)
    ref_sample = next(r.sample for r, _ in labeled if signature(r.sample) == ref_sig)
    ref = {"signature": ref_sig,
           "wavelengths": [float(w) for w in ref_sample.wavelengths] if ref_sig[0] == "wl" else None,
           "n_bands": ref_sample.n_bands,
           "feature_names": [ref_sample.band_label(i) for i in range(ref_sample.n_bands)]}

    X, y, groups = [], [], []
    for r, lbl in labeled:
        if r.spectrum is None or any(v is None for v in r.spectrum):
            info["incomplete"] += 1
            continue
        s = r.sample
        vec = align(r.spectrum, [float(w) for w in s.wavelengths] if s.wavelengths else None, signature(s), ref)
        if vec is None:
            info["incompatible"] += 1
            continue
        X.append(vec)
        y.append(lbl)
        groups.append(s.id)
    return np.array(X, dtype=float), np.array(y), np.array(groups), ref, info


def class_summary(y, groups):
    out = {}
    for lbl, g in zip(y, groups):
        d = out.setdefault(str(lbl), {"rois": 0, "samples": set()})
        d["rois"] += 1
        d["samples"].add(int(g))
    return {k: {"rois": v["rois"], "samples": len(v["samples"])} for k, v in sorted(out.items())}


def readiness(summary):
    if len(summary) < 2:
        return False, "Precisa de pelo menos 2 classes com ROIs medidos (ex.: 'verde' e 'normal')."
    min_rois = min(v["rois"] for v in summary.values())
    if min_rois < 2:
        return False, f"A classe com menos ROIs medidos só tem {min_rois} — precisa de pelo menos 2 por classe."
    return True, ""


def cv_plan(summary):
    """(modo, k): por amostra quando toda classe aparece em >= 2 amostras."""
    min_samples = min(v["samples"] for v in summary.values())
    if min_samples >= 2:
        return "group", min(5, min_samples)
    return "stratified", min(5, min(v["rois"] for v in summary.values()))


def _splitter(mode, k):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if mode == "group":
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


def estimate_seconds(X, y, algorithm, params, k):
    """Cronometra UM fit no dataset completo e multiplica pelos fits do treino (k + 1)."""
    pipe = make_pipeline(algorithm, params)   # fora do cronometro: inclui importar o sklearn na 1a vez
    t0 = time.perf_counter()
    pipe.fit(X, y)
    return max(time.perf_counter() - t0, 0.001) * (k + 1)


def train(X, y, groups, ref, algorithm, params, label_source, report=lambda f, m=None: None):
    """Valida (k-fold) e treina o modelo final com tudo. Devolve (bytes do joblib, info)."""
    import joblib
    from sklearn.metrics import confusion_matrix

    summary = class_summary(y, groups)
    ok, reason = readiness(summary)
    if not ok:
        raise MLError(reason)
    mode, k = cv_plan(summary)
    classes = sorted(summary)
    t0 = time.perf_counter()
    accuracies = []
    y_pred = np.empty_like(y)
    for i, (tr, te) in enumerate(_splitter(mode, k).split(X, y, groups)):
        report(0.05 + 0.8 * i / k, f"Validação cruzada: fold {i + 1} de {k}")
        pipe = make_pipeline(algorithm, params).fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        y_pred[te] = pred
        accuracies.append(float((pred == y[te]).mean()))
    cm = confusion_matrix(y, y_pred, labels=classes)
    report(0.88, "Treinando o modelo final com todos os dados")
    final = make_pipeline(algorithm, params).fit(X, y)
    train_seconds = time.perf_counter() - t0

    importance = None
    clf = final.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_).mean(axis=0)
    names = ref["feature_names"]
    feat = ([{"band": names[i], "wavelength": (ref["wavelengths"] or [None] * len(names))[i],
              "importance": float(v)} for i, v in enumerate(importance)] if importance is not None else None)

    info = {"algorithm": algorithm, "algorithm_label": ALGORITHMS[algorithm]["label"], "params": params,
            "label_source": label_source, "classes": classes, "class_summary": summary,
            "cv_mode": mode, "cv_folds": k,
            "cv_accuracy_mean": float(np.mean(accuracies)), "cv_accuracy_std": float(np.std(accuracies)),
            "confusion_matrix": cm.tolist(), "n_samples": int(len(X)), "n_features": int(X.shape[1]),
            "feature_importance": feat, "wavelengths": ref["wavelengths"],
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_seconds": round(train_seconds, 3)}
    buf = io.BytesIO()
    joblib.dump({"kind": "lambda-spectral", "pipeline": final, "reference": ref, "classes": classes,
                 "label_source": label_source}, buf)
    return buf.getvalue(), info


_model_cache = {}
_model_lock = threading.Lock()


def load_spectral_model(path):
    """Modelo espectral salvo por train(), ou None se o arquivo nao e um."""
    if not path or not path.endswith(".joblib") or not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
    import joblib
    obj = joblib.load(path)
    if not isinstance(obj, dict) or obj.get("kind") != "lambda-spectral":
        return None
    with _model_lock:
        _model_cache.clear()
        _model_cache[key] = obj
    return obj


def predict_spectra(model, sample, spectra):
    """[label ou ""] -- um por espectro (lista na ordem de bandas da amostra)."""
    wl = [float(w) for w in sample.wavelengths] if sample.wavelengths else None
    sig = signature(sample)
    vecs = [align(s, wl, sig, model["reference"]) for s in spectra]
    ok = [i for i, v in enumerate(vecs) if v is not None]
    out = [""] * len(spectra)
    if ok:
        preds = model["pipeline"].predict(np.array([vecs[i] for i in ok]))
        for i, p in zip(ok, preds):
            out[i] = str(p)
    return out
