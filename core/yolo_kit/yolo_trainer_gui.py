"""SpectralCrops -- Treinador YOLO (GUI)
==========================================

Ferramenta standalone que NAO faz parte do app Django e NAO roda no
Raspberry Pi (o Pi nao tem GPU, treinar aqui seria bem mais lento). Roda no
seu computador e treina um modelo YOLO de segmentacao ou deteccao a partir
do dataset .zip baixado em Annotations (botao "Exportar" -> Segmentacao ou
Deteccao) ou em Machine Learning.

Uso:
    1. pip install ultralytics
       (isso ja instala o PyTorch tambem; se voce tiver GPU NVIDIA, instale
       o PyTorch com suporte a CUDA primeiro pra treinar bem mais rapido --
       ver https://pytorch.org/get-started/locally/ -- e so depois
       "pip install ultralytics")
    2. python yolo_trainer_gui.py
    3. Extraia o .zip baixado do SpectralCrops numa pasta
    4. Clique em "Escolher data.yaml..." e selecione o data.yaml dessa pasta
    5. Escolha a tarefa (Segmentacao/Deteccao) -- tem que ser a MESMA do
       dataset que voce exportou -- ajuste os parametros se quiser (os
       padroes funcionam bem pra comecar) e clique em "Iniciar treinamento"
    6. Acompanhe o progresso na caixa de log. Ao final, clique em "Abrir
       pasta do modelo" pra achar o best.pt e suba ele de volta em
       Machine Learning, no SpectralCrops.
"""
import os
import re
import sys
import json
import math
import random
import shutil
import queue
import threading
import subprocess
import tkinter as tk

import yaml
import numpy as np
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime

try:
    import cv2   # vem junto com o ultralytics
except ImportError:
    cv2 = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


IMGSZ_OPTIONS = [320, 480, 640, 960]
MODEL_OPTIONS = ["n · nano", "s · small", "m · medium", "l · large"]
KIT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = KIT_DIR / "treinador_config.json"   # lembra a última pasta de saída
INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')   # caracteres proibidos em nome de pasta no Windows

# Parametros editaveis na aba "Parametros": (aba, chave do model.train, rotulo,
# tipo, padrao, opcoes, dica). Os padroes sao os do proprio ultralytics, entao
# sem mexer em nada o treino e o mesmo de sempre.
#   tipo "int"/"float": opcoes = (min, max, passo, formato)
#   tipo "combo": opcoes = lista de valores (so o 1o "pedaco" vai pro treino)
PARAMS = [
    ("basic", "model", "model", "combo", MODEL_OPTIONS[0], MODEL_OPTIONS,
     "Tamanho do modelo (da rede neural). n (nano) é o mais rápido e leve; s/m/l aprendem detalhes "
     "mais finos, mas treinam mais devagar, precisam de mais imagens e ficam mais lentos "
     "pra rodar no SpectralCrops. Com poucas imagens, fique no n ou s."),
    ("basic", "epochs", "epochs", "int", 100, (1, 2000, 10, "%.0f"),
     "Número de épocas. Quantas vezes o treino passa por todas as imagens. Mais épocas = mais aprendizado, "
     "até o ponto em que o modelo começa a decorar as imagens de treino (overfitting). "
     "Com dataset pequeno, 150–300 costuma ajudar."),
    ("basic", "imgsz", "imgsz", "combo", "640", [str(s) for s in IMGSZ_OPTIONS],
     "Tamanho da imagem. Resolução (em pixels) em que as imagens entram na rede. Maior preserva detalhes "
     "de sementes pequenas, mas usa mais memória e deixa treino e predição mais lentos."),
    ("basic", "batch", "batch", "int", -1, (-1, 128, 1, "%.0f"),
     "Tamanho do lote. Quantas imagens são processadas de uma vez. -1 escolhe sozinho pela memória da GPU. "
     "Se der erro de memória, diminua. Na CPU, valores pequenos (4–8) funcionam bem."),
    ("basic", "val_split", "val_split", "float", 0.2, (0.05, 0.5, 0.05, "%.2f"),
     "Proporção pra validação. Fração das amostras (ou das imagens, ver split_by) separada pra medir se o "
     "modelo aprendeu de verdade. 0.2 = 20% validação e 80% treino. Com poucas amostras, 0.15–0.25."),
    ("basic", "split_by", "split_by", "combo", "sample", ["sample", "image"],
     "Como dividir treino e validação. 'sample': por semente. Cada imagem entra nos dois lados, e as "
     "sementes do outro lado são apagadas (pintadas com a cor do fundo), então até 1–2 imagens funcionam. "
     "'image': cada imagem inteira vai pra um lado só. Mais rigoroso, mas precisa de várias imagens."),
    ("optim", "patience", "patience", "int", 100, (1, 1000, 10, "%.0f"),
     "Paciência (early stopping). Encerra o treino antes do fim se a métrica de validação não melhorar por esse número "
     "de épocas seguidas. Evita gastar tempo depois que o modelo estabilizou."),

    ("optim", "optimizer", "optimizer", "combo", "auto", ["auto", "SGD", "AdamW", "Adam"],
     "Otimizador. Algoritmo que ajusta os pesos da rede. 'auto' escolhe sozinho conforme o tamanho do "
     "treino (e ignora o lr0). AdamW costuma convergir rápido em "
     "datasets pequenos; SGD é o clássico, mais estável em treinos longos."),
    ("optim", "lr0", "lr0", "float", 0.01, (0.0001, 0.1, 0.001, "%.4f"),
     "Taxa de aprendizado inicial. Tamanho do passo de cada ajuste dos pesos. Alta demais: o treino oscila e não "
     "converge; baixa demais: aprende muito devagar. Só tem efeito quando o otimizador "
     "não é 'auto' (para AdamW, algo como 0.001)."),
    ("optim", "warmup_epochs", "warmup_epochs", "float", 3.0, (0, 20, 0.5, "%.1f"),
     "Épocas de aquecimento. Início do treino em que a taxa de aprendizado sobe aos poucos, pra não estragar os "
     "pesos pré-treinados. Atenção: o aquecimento dura no mínimo 100 lotes — com poucas "
     "imagens isso pode ser quase o treino inteiro. 0 desliga."),
    ("optim", "nbs", "nbs", "int", 64, (1, 256, 1, "%.0f"),
     "Lote nominal. Os pesos só são atualizados depois de acumular esse número de imagens. Com poucas "
     "imagens (ex.: 8 com o padrão 64), o modelo só atualiza a cada ~8 épocas. Deixar "
     "igual ao batch faz atualizar a cada lote."),
    ("optim", "cache", "cache", "bool", False, None,
     "Cache das imagens em RAM. Guarda as imagens na memória RAM em vez de reler do disco a cada época. Deixa o "
     "treino mais rápido; com datasets pequenos não tem custo."),

    ("aug", "mosaic", "mosaic", "float", 1.0, (0, 1, 0.1, "%.1f"),
     "Aumento por mosaico. Probabilidade de juntar 4 imagens em uma durante o treino. Ajuda o modelo a ver "
     "sementes em posições e escalas variadas. 0 desliga, 1 sempre."),
    ("aug", "close_mosaic", "close_mosaic", "int", 10, (0, 100, 1, "%.0f"),
     "Desliga o mosaico nas últimas N épocas. Épocas finais sem mosaico, pra o modelo terminar o treino vendo imagens "
     "parecidas com as reais."),
    ("aug", "fliplr", "fliplr", "float", 0.5, (0, 1, 0.1, "%.1f"),
     "Espelhamento horizontal. Probabilidade de espelhar cada imagem na horizontal. Sementes não têm lado, então "
     "isso aumenta a variedade sem custo."),
    ("aug", "flipud", "flipud", "float", 0.0, (0, 1, 0.1, "%.1f"),
     "Espelhamento vertical. Probabilidade de espelhar na vertical. O YOLO deixa em 0 (pensado para fotos com "
     "chão e céu), mas sementes vistas de cima podem usar 0.5."),
    ("aug", "degrees", "degrees", "float", 0.0, (0, 180, 5, "%.0f"),
     "Rotação máxima, em graus. Rotação aleatória de até ± esse ângulo. Útil quando as sementes podem aparecer em "
     "qualquer orientação na bandeja."),
    ("aug", "hsv_v", "hsv_v", "float", 0.4, (0, 1, 0.1, "%.1f"),
     "Variação aleatória de brilho (o V do HSV). Ajuda o modelo a tolerar diferenças de exposição e "
     "iluminação. Com iluminação bem controlada (LEDs fixos), pode diminuir."),
]
PARAM_TABS = [("basic", "Básico"), ("optim", "Otimização"), ("aug", "Aumento de dados")]
GPU_TIP = ("Treina na placa de vídeo NVIDIA (CUDA) — muito mais rápido que no processador. "
           "Vem marcada quando uma GPU é detectada. Desmarque só se a GPU der erro "
           "(ex.: falta de memória de vídeo); aí o treino roda na CPU.")
NO_GPU_TIP = ("Nenhuma GPU NVIDIA com CUDA foi detectada (ou o PyTorch instalado é só pra CPU), "
              "então o treino vai rodar no processador.")

# mesma paleta da interface web do SpectralCrops
C = {
    "bg": "#f4f6f8", "panel": "#ffffff", "line": "#e6e9ee",
    "ink": "#1f2937", "muted": "#6b7280", "muted2": "#9aa2ad",
    "blue": "#2f66e0", "blue_dark": "#2557c4", "blue_soft": "#eaf0fd",
    "green": "#2f9e5f", "green_soft": "#e3f6e9", "red": "#dc2626", "amber": "#b45309",
    "log_bg": "#0f1117", "log_fg": "#d1d5db",
}
FONT = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")  # cores/cursor que o tqdm do ultralytics manda pro terminal


class _QueueWriter:
    """Arquivo-like que junta o que o ultralytics escreve em stdout/stderr
    (incluindo as barras de progresso, que usam \\r sem \\n) e manda pra uma
    Queue como linhas completas -- ("line", texto) linha normal (fica no
    log), ("progress", texto) atualizacao de barra de progresso (sobrescreve
    a ultima linha em vez de acumular, senao o log vira um poema)."""

    def __init__(self, q):
        self.q = q
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += _ANSI_RE.sub("", s)
        while True:
            idx_n = self._buf.find("\n")
            idx_r = self._buf.find("\r")
            if idx_n == -1 and idx_r == -1:
                break
            if idx_n != -1 and (idx_r == -1 or idx_n < idx_r):
                line, self._buf = self._buf[:idx_n], self._buf[idx_n + 1:]
                self.q.put(("line", line))
            else:
                line, self._buf = self._buf[:idx_r], self._buf[idx_r + 1:]
                self.q.put(("progress", line))

    def flush(self):
        pass


def _batch_progress(trainer, q, state):
    state["batch"] += 1
    nb = len(trainer.train_loader)
    losses = "  ".join(f"{k.split('/')[-1]}={float(v):.3f}" for k, v in (trainer.tloss or {}).items())
    q.put(("progress", f"Época {trainer.epoch + 1}/{trainer.epochs} · lote {state['batch']}/{nb} · {losses}"))


def _epoch_summary(trainer, q):
    metrics = trainer.metrics or {}
    parts = [f"{k.split('/')[-1]}={metrics[k]:.3f}" for k in metrics if k.startswith("metrics/")]
    # depois da ultima epoca o ultralytics valida o best.pt e dispara esse
    # callback de novo, com epoch = epochs (por isso aparecia "Época 101/100")
    final = trainer.epoch + 1 > trainer.epochs
    title = "Validação final (best.pt)" if final else f"Época {trainer.epoch + 1}/{trainer.epochs} concluída"
    q.put(("line", f"{title} · " + "  ".join(parts)))
    q.put(("epoch", (min(trainer.epoch + 1, trainer.epochs), trainer.epochs, dict(metrics), final)))


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _label_path(img_path):
    """images/<split>/x.png -> labels/<split>/x.txt (convenção do YOLO)."""
    parts = list(Path(img_path).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_labels(img_path):
    lbl = _label_path(img_path)
    if not lbl.exists():
        return []
    return [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _read_dataset(yaml_path):
    """Lê um data.yaml exportado e devolve a lista de imagens (com as linhas
    de label de cada uma), os nomes das classes em ordem e as contagens. O
    export usa "path: ." relativo à pasta do yaml. A divisão treino/validação
    do export é ignorada: o treinador junta tudo e divide de novo
    (ver _prepare_split)."""
    src = Path(yaml_path).resolve()
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    root = (src.parent / cfg.get("path", ".")).resolve()
    names = cfg.get("names") or {}
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    dirs = []
    for split in ("train", "val"):
        val = cfg.get(split) or []
        dirs += [(root / d).resolve() for d in ([val] if isinstance(val, str) else val)]
    images = []
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    images.append({"img": f, "lines": _read_labels(f)})
    return {"yaml": src, "names": names, "train": [str(d) for d in dirs], "images": images,
            "n_images": len(images), "n_samples": sum(len(i["lines"]) for i in images)}


def _shape_mask(lines, idxs, h, w):
    """Máscara (uint8) com as formas `idxs` pintadas -- polígono (seg) ou caixa (det)."""
    m = np.zeros((h, w), np.uint8)
    for i in idxs:
        c = [float(v) for v in lines[i][1:]]
        if len(c) == 4:   # det: cx cy w h normalizados
            cx, cy, bw, bh = c
            cv2.rectangle(m, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int(math.ceil((cx + bw / 2) * w)), int(math.ceil((cy + bh / 2) * h))), 255, -1)
        elif len(c) >= 6:   # seg: x1 y1 x2 y2 ... normalizados
            pts = np.round(np.array(c).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(m, [pts], 255)
    return m


def _hide_samples(img, lines, hide_idx):
    """Apaga as amostras `hide_idx` da imagem pintando cada uma com a cor do
    fundo AO REDOR dela (mediana de um anel em volta, fora de qualquer
    anotação) -- a iluminação varia pela bandeja, então uma cor única pro fundo
    todo ficaria destoando. Uma semente que aparece na imagem SEM anotação
    viraria exemplo de "fundo" pro YOLO -- por isso a que vai pra validação
    some da cópia de treino, e vice-versa."""
    h, w = img.shape[:2]
    r = max(3, int(0.004 * min(h, w)))   # folga pra cobrir a borda/sombra da semente
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring_r = max(8, int(0.025 * min(h, w)))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_r + 1, 2 * ring_r + 1))
    hide_set = set(hide_idx)
    keep = _shape_mask(lines, [i for i in range(len(lines)) if i not in hide_set], h, w)
    all_shapes = cv2.dilate(_shape_mask(lines, range(len(lines)), h, w), kernel)
    bg_all = img[all_shapes == 0]
    fallback = np.median(bg_all if len(bg_all) else img.reshape(-1, *img.shape[2:]), axis=0)
    out = img.copy()
    for i in hide_idx:
        one = _shape_mask(lines, [i], h, w)
        region = cv2.dilate(one, kernel)
        region[keep > 0] = 0                       # a folga nunca come uma semente vizinha
        ring = (cv2.dilate(one, ring_kernel) > 0) & (all_shapes == 0)
        color = np.median(img[ring], axis=0) if ring.sum() >= 20 else fallback
        out[region > 0] = color.astype(img.dtype)
    return out


def _imread(path):   # cv2.imread não abre caminho com acento no Windows
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def _imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise OSError(f"não consegui gravar {path}")
    buf.tofile(str(path))


def _prepare_split(datasets, split_by, val_frac, log, out_dir):
    """Junta as imagens de todos os datasets e divide treino/validação de novo,
    gravando em <out_dir>/_split_treino/ (apagada e recriada a cada treino).
    Devolve o caminho do data_treino.yaml (também em out_dir).

    split_by="sample": cada imagem contribui ~val_frac das SUAS amostras pra
      validação. Viram duas cópias da imagem: a de treino sem as amostras de
      validação e a de validação sem as de treino (ver _hide_samples). Assim
      até 1-2 imagens rendem treino e validação, e cada semente fica em um só lado.
    split_by="image": a imagem inteira vai pra um lado (precisa de 2+ imagens).

    `log(texto)` vira linha no log; `log(("progress", texto))` sobrescreve a última."""
    out_dir = Path(out_dir)
    split_dir = out_dir / "_split_treino"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (split_dir / sub).mkdir(parents=True)

    items, used = [], set()
    for di, ds in enumerate(datasets):
        for it in ds["images"]:
            name = f"d{di + 1}_{it['img'].stem}" if len(datasets) > 1 else it["img"].stem
            while name in used:      # mesmo nome em train/ e val/ do export
                name += "_"
            used.add(name)
            items.append({**it, "name": name})

    def put(split, it, idxs, hide_idx):
        dst = split_dir / "images" / split / (it["name"] + it["img"].suffix)
        if hide_idx:
            _imwrite(dst, _hide_samples(_imread(it["img"]), it["lines"], hide_idx))
        else:
            shutil.copy2(it["img"], dst)
        (split_dir / "labels" / split / (it["name"] + ".txt")).write_text(
            "\n".join(" ".join(it["lines"][i]) for i in idxs), encoding="utf-8")

    rng = random.Random(42)   # semente fixa: a mesma divisão a cada treino
    if split_by == "image":
        if len(items) < 2:
            raise ValueError("Com split_by = image precisa de pelo menos 2 imagens. Use split_by = sample.")
        order = list(range(len(items)))
        rng.shuffle(order)
        val_set = set(order[:max(1, round(len(items) * val_frac))])
        if len(val_set) == len(items):
            val_set.pop()
        for k, it in enumerate(items):
            put("val" if k in val_set else "train", it, list(range(len(it["lines"]))), [])
        n_val_img = len(val_set)
        log(f"Divisão por imagem: {len(items) - n_val_img} imagens de treino · {n_val_img} de validação "
            f"({sum(len(i['lines']) for i in items)} amostras no total).")
    else:
        val_idx = []
        for it in items:
            idx = list(range(len(it["lines"])))
            rng.shuffle(idx)
            val_idx.append(sorted(idx[:round(len(idx) * val_frac)]))
        if not any(val_idx):     # poucas amostras: garante ao menos 1 na validação
            k = max(range(len(items)), key=lambda j: len(items[j]["lines"]))
            if len(items[k]["lines"]) < 2:
                raise ValueError("Amostras insuficientes pra separar treino e validação.")
            val_idx[k] = [rng.randrange(len(items[k]["lines"]))]
        n_tr = n_va = 0
        for k, (it, vi) in enumerate(zip(items, val_idx)):
            vi_set = set(vi)
            ti = [i for i in range(len(it["lines"])) if i not in vi_set]
            log(("progress", f"Preparando imagens… {k + 1}/{len(items)}"))
            if ti or not it["lines"]:        # imagem sem amostra = exemplo de fundo, vai pro treino
                put("train", it, ti, vi)
            if vi:
                put("val", it, vi, ti)
            n_tr += len(ti)
            n_va += len(vi)
        log(f"Divisão por amostra: {n_tr} amostras de treino · {n_va} de validação "
            f"({n_va / max(1, n_tr + n_va):.0%}), em {len(items)} imagens.")

    cfg = {"path": str(split_dir), "train": "images/train", "val": "images/val",
           "names": dict(enumerate(datasets[0]["names"])),
           "fontes": [str(ds["yaml"]) for ds in datasets],   # só referência, o YOLO ignora
           "split_by": split_by, "val_split": val_frac}
    dst = out_dir / "data_treino.yaml"
    dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(dst)


class _Tooltip:
    """Balão de dica que aparece ao passar o mouse sobre um widget."""

    def __init__(self, widget, text):
        self.widget, self.text = widget, text
        self.win, self._after = None, None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._after = self.widget.after(250, self._show)

    def _show(self):
        if self.win:
            return
        self.win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tk.Label(tw, text=self.text, bg=C["ink"], fg="#ffffff", font=(FONT, 9), wraplength=300,
                 justify="left", padx=10, pady=7).pack()
        tw.update_idletasks()
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
        if x + tw.winfo_width() > tw.winfo_screenwidth():
            x = self.widget.winfo_rootx() - tw.winfo_width() - 8
        tw.wm_geometry(f"+{x}+{self.widget.winfo_rooty() - 6}")

    def _hide(self, _e=None):
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None
        if self.win:
            self.win.destroy()
            self.win = None


class _OutputDialog(tk.Toplevel):
    """Pergunta a pasta-mãe e o nome da pasta de saída do treino. `result` fica
    com o Path final (pasta-mãe / nome) ou None se cancelar."""

    def __init__(self, parent, parent_dir, name):
        super().__init__(parent)
        self.title("Onde salvar o treino")
        self.configure(bg=C["panel"])
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        self.parent_dir = tk.StringVar(value=parent_dir)
        self.name = tk.StringVar(value=name)
        self.preview = tk.StringVar()
        self.error = tk.StringVar()

        body = ttk.Frame(self, style="Card.TFrame", padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Onde salvar o treino", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(body, text="O modelo (weights/best.pt), os gráficos e as imagens usadas ficam todos nessa pasta.",
                  style="Muted.TLabel", wraplength=460).pack(anchor="w", pady=(2, 14))

        ttk.Label(body, text="Salvar em", style="Card.TLabel").pack(anchor="w")
        row = ttk.Frame(body, style="Card.TFrame")
        row.pack(fill="x", pady=(4, 12))
        ttk.Entry(row, textvariable=self.parent_dir, width=52).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Escolher…", style="Secondary.TButton",
                   command=self._browse).pack(side="left", padx=(8, 0))

        ttk.Label(body, text="Nome da pasta", style="Card.TLabel").pack(anchor="w")
        name_entry = ttk.Entry(body, textvariable=self.name)
        name_entry.pack(fill="x", pady=(4, 10))

        ttk.Label(body, textvariable=self.preview, style="Muted.TLabel", wraplength=460).pack(anchor="w")
        tk.Label(body, textvariable=self.error, bg=C["panel"], fg=C["red"], font=(FONT, 9),
                 wraplength=460, justify="left").pack(anchor="w", pady=(4, 0))

        btns = ttk.Frame(body, style="Card.TFrame")
        btns.pack(fill="x", pady=(14, 0))
        ttk.Button(btns, text="Iniciar treinamento", style="Primary.TButton",
                   command=self._ok).pack(side="right")
        ttk.Button(btns, text="Cancelar", style="Secondary.TButton",
                   command=self.destroy).pack(side="right", padx=(0, 8))

        self.parent_dir.trace_add("write", lambda *a: self._refresh())
        self.name.trace_add("write", lambda *a: self._refresh())
        self._refresh()
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())

        self.update_idletasks()   # centraliza sobre a janela principal
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.grab_set()
        name_entry.focus_set()
        name_entry.select_range(0, "end")

    def _target(self):
        return Path(self.parent_dir.get().strip()) / self.name.get().strip()

    def _problem(self):
        name, parent = self.name.get().strip(), self.parent_dir.get().strip()
        if not parent:
            return "Escolha onde salvar."
        if not Path(parent).is_dir():
            return "A pasta escolhida em “Salvar em” não existe."
        if not name:
            return "Dê um nome pra pasta."
        if INVALID_NAME_RE.search(name) or name in (".", ".."):
            return 'O nome não pode ter estes caracteres: < > : " / \\ | ? *'
        return None

    def _refresh(self):
        problem = self._problem()
        self.error.set(problem or "")
        if problem:
            self.preview.set("")
            return
        target = self._target()
        note = "  (já existe, os arquivos de mesmo nome serão substituídos)" if target.exists() else ""
        self.preview.set(f"Pasta do treino: {target}{note}")

    def _browse(self):
        start = self.parent_dir.get() if Path(self.parent_dir.get() or ".").is_dir() else str(Path.home())
        path = filedialog.askdirectory(parent=self, title="Onde salvar o treino", initialdir=start)
        if path:
            self.parent_dir.set(str(Path(path)))

    def _ok(self):
        if self._problem():
            return
        target = self._target()
        if target.exists() and any(target.iterdir()):
            if not messagebox.askyesno(
                    "Pasta já existe",
                    f"A pasta “{target.name}” já existe e não está vazia.\n\n"
                    "Os arquivos do treino anterior com o mesmo nome serão substituídos. Continuar?",
                    parent=self):
                return
        self.result = target
        self.destroy()


class _RedirectOutput:
    """Context manager que redireciona stdout/stderr pra uma Queue -- ver
    _QueueWriter. Usado so durante o treino, num thread separado, pra GUI
    mostrar o log em tempo real sem travar a interface principal."""

    def __init__(self, q):
        self.writer = _QueueWriter(q)
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.writer, self.writer
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
        return False


class TrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SpectralCrops — Treinador YOLO")
        self.geometry("1040x780")
        self.minsize(900, 740)
        self.configure(bg=C["bg"])

        self.datasets = []   # resultado de _read_dataset() de cada data.yaml adicionado
        self.task = tk.StringVar(value="seg")
        var_types = {"int": tk.IntVar, "float": tk.DoubleVar, "combo": tk.StringVar, "bool": tk.BooleanVar}
        self.params = {key: var_types[kind](value=default) for _, key, _, kind, default, _, _ in PARAMS}
        self.use_gpu = tk.BooleanVar(value=False)   # marcada em _check_deps() se tiver CUDA
        self._has_gpu = False
        self.status_text = tk.StringVar(value="Selecione o data.yaml pra começar.")
        self.epoch_text = tk.StringVar(value="—")

        self._log_queue = queue.Queue()
        self._training_thread = None
        self._last_save_dir = None
        self._last_line_was_progress = False
        self._images = {}   # referencias dos PhotoImage (senao o Tk descarta)

        self._setup_style()
        self._build_ui()
        self._check_deps()
        self.after(150, self._poll_log_queue)

    # ── Estilo ──────────────────────────────────────────────────────────
    def _setup_style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=C["bg"], foreground=C["ink"], font=(FONT, 10))
        st.configure("Card.TFrame", background=C["panel"])
        st.configure("Card.TLabel", background=C["panel"], foreground=C["ink"])
        st.configure("Muted.TLabel", background=C["panel"], foreground=C["muted"], font=(FONT, 9))
        st.configure("CardTitle.TLabel", background=C["panel"], foreground=C["ink"], font=(FONT, 11, "bold"))
        st.configure("Step.TLabel", background=C["blue_soft"], foreground=C["blue"], font=(FONT, 9, "bold"),
                     padding=(7, 1))
        st.configure("Status.TLabel", background=C["bg"], foreground=C["muted"], font=(FONT, 10))
        st.configure("Card.TCheckbutton", background=C["panel"], foreground=C["ink"])
        st.map("Card.TCheckbutton", background=[("active", C["panel"])])
        st.configure("TEntry", fieldbackground=C["bg"], bordercolor=C["line"], lightcolor=C["line"],
                     darkcolor=C["line"], padding=6)
        st.configure("TSpinbox", fieldbackground=C["panel"], bordercolor=C["line"], lightcolor=C["line"],
                     darkcolor=C["line"], arrowsize=12, padding=4)
        st.configure("TCombobox", fieldbackground=C["panel"], bordercolor=C["line"], lightcolor=C["line"],
                     darkcolor=C["line"], arrowsize=12, padding=4)
        # cores de seleção legíveis: no clam, o combobox readonly com foco pintava o
        # texto selecionado de branco sobre o fundo branco (o valor "sumia")
        st.map("TCombobox", fieldbackground=[("readonly", C["panel"])],
               selectbackground=[("readonly", C["panel"]), ("focus", C["blue_soft"])],
               selectforeground=[("readonly", C["ink"]), ("focus", C["ink"])])
        st.map("TSpinbox", selectbackground=[("focus", C["blue_soft"])],
               selectforeground=[("focus", C["ink"])])
        st.configure("TEntry", selectbackground=C["blue_soft"], selectforeground=C["ink"])

        st.configure("Primary.TButton", background=C["blue"], foreground="#ffffff", bordercolor=C["blue"],
                     lightcolor=C["blue"], darkcolor=C["blue"], font=(FONT, 11, "bold"), padding=(18, 10))
        st.map("Primary.TButton",
               background=[("disabled", "#a9bdf0"), ("pressed", C["blue_dark"]), ("active", C["blue_dark"])],
               bordercolor=[("disabled", "#a9bdf0"), ("active", C["blue_dark"])],
               lightcolor=[("disabled", "#a9bdf0"), ("active", C["blue_dark"])],
               darkcolor=[("disabled", "#a9bdf0"), ("active", C["blue_dark"])],
               foreground=[("disabled", "#eef2fc")])
        st.configure("Secondary.TButton", background=C["panel"], foreground=C["ink"], bordercolor=C["line"],
                     lightcolor=C["panel"], darkcolor=C["panel"], padding=(14, 8))
        st.map("Secondary.TButton",
               background=[("disabled", C["panel"]), ("active", C["blue_soft"])],
               foreground=[("disabled", C["muted2"]), ("active", C["blue"])],
               bordercolor=[("active", C["blue"])])

        st.configure("Tip.TLabel", background=C["panel"], foreground=C["muted2"], font=(FONT, 10))
        st.map("Tip.TLabel", foreground=[("active", C["blue"])])
        st.configure("Link.TLabel", background=C["panel"], foreground=C["blue"], font=(FONT, 9))
        st.configure("TNotebook", background=C["panel"], borderwidth=0, tabmargins=(0, 0, 0, 0))
        st.configure("TNotebook.Tab", background=C["bg"], foreground=C["muted"], bordercolor=C["line"],
                     lightcolor=C["line"], padding=(10, 4), font=(FONT, 9),
                     focuscolor=C["panel"])
        st.map("TNotebook.Tab", background=[("selected", C["panel"])],
               foreground=[("selected", C["blue"])], font=[("selected", (FONT, 9, "bold"))],
               lightcolor=[("selected", C["panel"])])

        st.configure("Epoch.Horizontal.TProgressbar", troughcolor=C["line"], background=C["green"],
                     bordercolor=C["line"], lightcolor=C["green"], darkcolor=C["green"], thickness=10)

    def _load_image(self, name):
        try:
            img = tk.PhotoImage(file=str(KIT_DIR / name))
        except tk.TclError:
            return None   # kit antigo sem o png: segue sem imagem
        self._images[name] = img
        return img

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        icon = self._load_image("icon.png")
        if icon:
            self.iconphoto(True, icon)

        # cabeçalho
        header = tk.Frame(self, bg=C["panel"], highlightthickness=0)
        header.pack(fill="x")
        tk.Frame(self, bg=C["line"], height=1).pack(fill="x")
        logo = self._load_image("logo.png")
        if logo:
            tk.Label(header, image=logo, bg=C["panel"]).pack(side="left", padx=(20, 12), pady=12)
        titles = tk.Frame(header, bg=C["panel"])
        titles.pack(side="left", pady=12, padx=(0 if logo else 20, 0))
        tk.Label(titles, text="SpectralCrops", bg=C["panel"], fg=C["ink"],
                 font=(FONT, 17, "bold")).pack(anchor="w")
        tk.Label(titles, text="Treinador de modelos YOLO", bg=C["panel"], fg=C["muted"],
                 font=(FONT, 10)).pack(anchor="w")
        self.device_badge = tk.Label(header, text="Verificando GPU…", bg=C["bg"], fg=C["muted"],
                                     font=(FONT, 9, "bold"), padx=10, pady=4)
        self.device_badge.pack(side="right", padx=20)

        # corpo: configuração (esq.) | progresso (dir.)
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=16)
        left = tk.Frame(body, bg=C["bg"], width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self._build_config(left)
        self._build_progress(right)

    def _tip_icon(self, parent, text):
        icon = ttk.Label(parent, text="ⓘ", style="Tip.TLabel", cursor="question_arrow")
        icon.bind("<Enter>", lambda e: icon.state(["active"]), add="+")
        icon.bind("<Leave>", lambda e: icon.state(["!active"]), add="+")
        _Tooltip(icon, text)
        return icon

    def _card(self, parent, step, title, tip=None, expand=False):
        outer = tk.Frame(parent, bg=C["line"])   # borda de 1px
        outer.pack(fill="both" if expand else "x", expand=expand, pady=(0, 12))
        card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True, padx=1, pady=1)
        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 10))
        if step:
            ttk.Label(head, text=step, style="Step.TLabel").pack(side="left", padx=(0, 8))
        ttk.Label(head, text=title, style="CardTitle.TLabel").pack(side="left")
        if tip:
            self._tip_icon(head, tip).pack(side="left", padx=(6, 0))
        self._card_head = head   # pra quem quiser pôr algo à direita do título
        return card

    def _build_config(self, parent):
        # 1. dataset
        card = self._card(parent, "1", "Datasets",
                          "Adicione o data.yaml da pasta extraída do .zip exportado no SpectralCrops "
                          "(Annotations → Exportar, ou Machine Learning). Dá pra adicionar exports de "
                          "projetos diferentes e treinar tudo junto — desde que tenham as mesmas classes. "
                          "Nada é copiado: o treino lê as imagens direto de cada pasta.")
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        box = tk.Frame(row, bg=C["line"])
        box.pack(side="left", fill="both", expand=True)
        self.ds_list = tk.Listbox(box, height=3, relief="flat", bd=0, highlightthickness=0, activestyle="none",
                                  bg=C["bg"], fg=C["ink"], font=(FONT, 9), selectmode="extended",
                                  selectbackground=C["blue_soft"], selectforeground=C["blue"])
        self.ds_list.pack(fill="both", expand=True, padx=1, pady=1)
        btns = ttk.Frame(row, style="Card.TFrame")
        btns.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(btns, text="Adicionar…", style="Secondary.TButton",
                   command=self._pick_data_yaml).pack(fill="x")
        ttk.Button(btns, text="Remover", style="Secondary.TButton",
                   command=self._remove_dataset).pack(fill="x", pady=(6, 0))

        # 2. tarefa (botões de alternância)
        card = self._card(parent, "2", "Tarefa",
                          "Tem que ser a mesma do dataset exportado. Segmentação aprende o contorno "
                          "de cada semente (polígono); Detecção aprende só a caixa ao redor.")
        seg = tk.Frame(card, bg=C["line"])
        seg.pack(fill="x")
        self._task_btns = {}
        for value, text in (("seg", "Segmentação  (contorno)"), ("det", "Detecção  (caixa)")):
            b = tk.Radiobutton(seg, text=text, variable=self.task, value=value, indicatoron=False,
                               font=(FONT, 10), relief="flat", bd=0, pady=7, cursor="hand2",
                               command=self._refresh_task_buttons)
            b.pack(side="left", fill="x", expand=True, padx=1, pady=1)
            self._task_btns[value] = b
        self._refresh_task_buttons()

        # 3. parâmetros
        card = self._card(parent, "3", "Parâmetros",
                          "Os valores iniciais são os padrões do YOLO. Passe o mouse no ⓘ de cada "
                          "parâmetro pra ver o que ele muda no treino.")
        reset = ttk.Label(self._card_head, text="Restaurar padrões", style="Link.TLabel", cursor="hand2")
        reset.pack(side="right")
        reset.bind("<Button-1>", lambda e: self._reset_params())

        nb = ttk.Notebook(card)
        # o Notebook passa o foco pro 1º campo da aba ao trocar; mantém o foco na aba
        nb.bind("<<NotebookTabChanged>>", lambda e: nb.focus_set())
        nb.pack(fill="x")
        tabs = {}
        for tab_key, tab_title in PARAM_TABS:
            frame = ttk.Frame(nb, style="Card.TFrame", padding=(2, 8, 2, 2))
            frame.columnconfigure(0, weight=1)
            nb.add(frame, text=tab_title)
            tabs[tab_key] = frame

        for tab_key, key, label, kind, _default, opts, tip in PARAMS:
            frame = tabs[tab_key]
            r = frame.grid_size()[1]
            if kind == "bool":
                lbl = ttk.Frame(frame, style="Card.TFrame")
                ttk.Checkbutton(lbl, text=label, variable=self.params[key],
                                style="Card.TCheckbutton").pack(side="left")
                self._tip_icon(lbl, tip).pack(side="left", padx=(4, 0))
                lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
                continue
            lbl = ttk.Frame(frame, style="Card.TFrame")
            ttk.Label(lbl, text=label, style="Card.TLabel").pack(side="left")
            self._tip_icon(lbl, tip).pack(side="left", padx=(5, 0))
            lbl.grid(row=r, column=0, sticky="w", pady=3)
            if kind == "combo":
                w = ttk.Combobox(frame, textvariable=self.params[key], values=opts, state="readonly",
                                 width=11 if key == "model" else 8)
            else:
                lo, hi, step, fmt = opts
                w = ttk.Spinbox(frame, from_=lo, to=hi, increment=step, format=fmt,
                                textvariable=self.params[key], width=8)
            w.grid(row=r, column=1, sticky="e", pady=3)

        cpu = ttk.Frame(tabs["basic"], style="Card.TFrame")
        self.gpu_check = ttk.Checkbutton(cpu, text="Use GPU", variable=self.use_gpu,
                                         style="Card.TCheckbutton", state="disabled")
        self.gpu_check.pack(side="left")
        self._gpu_tip_frame = cpu
        cpu.grid(row=tabs["basic"].grid_size()[1], column=0, columnspan=2, sticky="w", pady=4)

        # ações
        actions = tk.Frame(parent, bg=C["bg"])
        actions.pack(fill="x", side="bottom")
        self.btn_train = ttk.Button(actions, text="Iniciar treinamento", style="Primary.TButton",
                                    command=self._start_training)
        self.btn_open_folder = ttk.Button(actions, text="Abrir pasta do modelo",
                                          style="Secondary.TButton", command=self._open_result_folder,
                                          state="disabled")
        self.btn_open_folder.pack(side="right", fill="y", padx=(8, 0))
        self.btn_train.pack(side="left", fill="x", expand=True)

    def _refresh_task_buttons(self):
        for value, b in self._task_btns.items():
            on = self.task.get() == value
            b.configure(bg=C["blue_soft"] if on else C["panel"], fg=C["blue"] if on else C["muted"],
                        selectcolor=C["blue_soft"], activebackground=C["blue_soft"],
                        activeforeground=C["blue"], font=(FONT, 10, "bold" if on else "normal"))

    def _build_progress(self, parent):
        card = self._card(parent, None, "Progresso", expand=True)

        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.status_text, style="Card.TLabel").pack(side="left")
        ttk.Label(top, textvariable=self.epoch_text, style="Muted.TLabel").pack(side="right")
        self.progress = ttk.Progressbar(card, style="Epoch.Horizontal.TProgressbar", mode="determinate")
        self.progress.pack(fill="x", pady=(6, 12))

        # painel de métricas da última validação
        tiles = tk.Frame(card, bg=C["panel"])
        tiles.pack(fill="x", pady=(0, 12))
        self._metric_vals = {}
        for i, (key, label) in enumerate((("mAP50", "mAP50"), ("mAP50-95", "mAP50-95"),
                                           ("precision", "Precisão"), ("recall", "Recall"))):
            tiles.columnconfigure(i, weight=1, uniform="tile")
            t = tk.Frame(tiles, bg=C["bg"], padx=12, pady=8)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            tk.Label(t, text=label, bg=C["bg"], fg=C["muted"], font=(FONT, 9)).pack(anchor="w")
            v = tk.Label(t, text="—", bg=C["bg"], fg=C["ink"], font=(FONT, 16, "bold"))
            v.pack(anchor="w")
            self._metric_vals[key] = v

        log_wrap = tk.Frame(card, bg=C["log_bg"])
        log_wrap.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_wrap, height=14, wrap="word", state="disabled", relief="flat", bd=0,
                                bg=C["log_bg"], fg=C["log_fg"], insertbackground=C["log_fg"],
                                font=("Consolas" if sys.platform.startswith("win") else "Monospace", 9),
                                padx=10, pady=8)
        sb = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def _set_device(self, text, ok):
        fg, bg = (C["green"], C["green_soft"]) if ok else (C["amber"], "#fdf1e4")
        self.device_badge.config(text=text, fg=fg, bg=bg)

    # ── Checagens ───────────────────────────────────────────────────────
    def _check_deps(self):
        if not HAS_ULTRALYTICS:
            messagebox.showerror(
                "Dependência faltando",
                "O pacote 'ultralytics' não está instalado nesse Python.\n\n"
                "Abra um terminal e rode:\n    pip install ultralytics\n\n"
                "e reabra esse programa.")
            self.btn_train.config(state="disabled")
            self.device_badge.config(text="ultralytics não instalado", fg=C["red"], bg="#fdeaea")
            return
        self._has_gpu = HAS_TORCH and torch.cuda.is_available()
        if self._has_gpu:
            name = torch.cuda.get_device_name(0)
            self._set_device(f"GPU: {name}", ok=True)
            self.use_gpu.set(True)
            self.gpu_check.config(state="normal")
            self._tip_icon(self._gpu_tip_frame, f"{GPU_TIP}\n\nDetectada: {name}").pack(side="left", padx=(4, 0))
        else:
            self._set_device("Sem GPU CUDA — treino na CPU (mais lento)", ok=False)
            self._tip_icon(self._gpu_tip_frame, NO_GPU_TIP).pack(side="left", padx=(4, 0))

    # ── Ações ───────────────────────────────────────────────────────────
    def _pick_data_yaml(self):
        path = filedialog.askopenfilename(
            title="Selecione o data.yaml do dataset exportado",
            filetypes=[("data.yaml", "data.yaml"), ("YAML", "*.yaml;*.yml"), ("Todos os arquivos", "*.*")])
        if not path:
            return
        try:
            ds = _read_dataset(path)
        except Exception as e:
            messagebox.showerror("Dataset inválido", f"Não consegui ler esse data.yaml:\n\n{e}")
            return
        if any(d["yaml"] == ds["yaml"] for d in self.datasets):
            return
        if not ds["names"] or not ds["images"]:
            messagebox.showerror("Dataset inválido", "Esse data.yaml não tem 'names' ou nenhuma imagem.")
            return
        if self.datasets and ds["names"] != self.datasets[0]["names"]:
            messagebox.showerror(
                "Classes diferentes",
                "Pra treinar junto, os datasets precisam ter as mesmas classes, na mesma ordem.\n\n"
                f"Já adicionado: {', '.join(self.datasets[0]['names'])}\n"
                f"Este: {', '.join(ds['names'])}\n\n"
                "Ajuste as categorias nos projetos do SpectralCrops e exporte de novo.")
            return
        self.datasets.append(ds)
        self._refresh_datasets()

    def _remove_dataset(self):
        for i in reversed(self.ds_list.curselection()):
            del self.datasets[i]
        self._refresh_datasets()

    def _refresh_datasets(self):
        self.ds_list.delete(0, "end")
        for ds in self.datasets:
            self.ds_list.insert("end", f"{ds['yaml'].parent.name}   ({ds['n_images']} imagens · {ds['n_samples']} amostras)")
        if not self.datasets:
            self.status_text.set("Selecione o data.yaml pra começar.")
            return
        n_img = sum(d["n_images"] for d in self.datasets)
        n_smp = sum(d["n_samples"] for d in self.datasets)
        what = "1 dataset" if len(self.datasets) == 1 else f"{len(self.datasets)} datasets combinados"
        self.status_text.set(f"{what} · {n_img} imagens · {n_smp} amostras")

    def _start_training(self):
        if not self.datasets:
            messagebox.showwarning("Falta o dataset", "Adicione pelo menos um data.yaml.")
            return
        if self._training_thread and self._training_thread.is_alive():
            return
        params = self._collect_params()
        if params is None:
            return
        out_dir = self._ask_output_dir()
        if out_dir is None:
            return
        self.btn_train.config(state="disabled", text="Treinando…")
        self.btn_open_folder.config(state="disabled")
        self.status_text.set("Treinando…")
        self.epoch_text.set(f"Época 0/{params['epochs']}")
        self.progress.configure(maximum=params["epochs"], value=0)
        for v in self._metric_vals.values():
            v.config(text="—")
        self._clear_log()
        # lido aqui (thread principal): variaveis do Tk nao devem ser lidas do thread de treino
        run_args = (params, self.task.get(), self.use_gpu.get(), list(self.datasets), out_dir)
        self._training_thread = threading.Thread(target=self._run_training, args=run_args, daemon=True)
        self._training_thread.start()

    def _ask_output_dir(self):
        """Mostra o diálogo de pasta de saída. Sugere a última pasta-mãe usada
        (ou a pasta-mãe do 1º dataset) e um nome com tarefa + data/hora."""
        cfg = self._load_config()
        parent_dir = cfg.get("output_parent")
        if not parent_dir or not Path(parent_dir).is_dir():
            parent_dir = str(self.datasets[0]["yaml"].parent.parent)
        name = f"treino_{self.task.get()}_{datetime.now():%Y-%m-%d_%H-%M}"
        dlg = _OutputDialog(self, parent_dir, name)
        self.wait_window(dlg)
        if dlg.result is not None:
            cfg["output_parent"] = str(dlg.result.parent)
            self._save_config(cfg)
        return dlg.result

    @staticmethod
    def _load_config():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_config(cfg):
        try:
            CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass   # sem permissão na pasta do kit: só não lembra da próxima vez

    def _collect_params(self):
        """Lê os campos da aba Parâmetros; None (com aviso) se algum estiver inválido."""
        params = {}
        for _, key, label, kind, _default, opts, _ in PARAMS:
            try:
                value = self.params[key].get()
            except tk.TclError:
                messagebox.showwarning("Parâmetro inválido", f"Valor inválido em “{label}”.")
                return None
            if kind == "combo":
                value = value.split(" ")[0]
            elif kind in ("int", "float"):
                lo, hi = opts[0], opts[1]
                if not lo <= value <= hi:
                    messagebox.showwarning("Parâmetro inválido",
                                           f"“{label}” tem que estar entre {lo:g} e {hi:g}.")
                    return None
            params[key] = value
        params["imgsz"] = int(params["imgsz"])
        return params

    def _reset_params(self):
        for _, key, _, _, default, _, _ in PARAMS:
            self.params[key].set(default)
        self.use_gpu.set(self._has_gpu)

    def _run_training(self, params, task, use_gpu, datasets, out_dir):
        try:
            params = dict(params)
            size = params.pop("model")
            split_by, val_frac = params.pop("split_by"), params.pop("val_split")   # não são do model.train

            def log(msg):
                self._log_queue.put(msg if isinstance(msg, tuple) else ("line", msg))
            out_dir.mkdir(parents=True, exist_ok=True)
            log(f"Pasta do treino: {out_dir}")
            data_yaml = _prepare_split(datasets, split_by, val_frac, log, out_dir)
            checkpoint = f"yolov8{size}-seg.pt" if task == "seg" else f"yolov8{size}.pt"
            device = 0 if use_gpu else "cpu"
            self._log_queue.put(("line", f"Dataset: {data_yaml}"))
            self._log_queue.put(("line", f"Modelo base: {checkpoint} · " +
                                 "  ".join(f"{k}={v}" for k, v in params.items())))

            with _RedirectOutput(self._log_queue):
                model = YOLO(checkpoint)
                state = {"batch": 0}
                model.add_callback("on_train_epoch_start", lambda t: state.update(batch=0))
                model.add_callback("on_train_batch_end", lambda t: _batch_progress(t, self._log_queue, state))
                model.add_callback("on_fit_epoch_end", lambda t: _epoch_summary(t, self._log_queue))
                results = model.train(
                    data=data_yaml,
                    device=device,
                    **params,
                    project=str(out_dir.parent),   # salva direto em out_dir (sem runs/…/ no meio)
                    name=out_dir.name,
                    exist_ok=True,
                )

            save_dir = (getattr(results, "save_dir", None)
                        or getattr(getattr(model, "trainer", None), "save_dir", None)
                        or out_dir)
            save_dir = Path(save_dir)
            best = save_dir / "weights" / "best.pt"
            self._last_save_dir = best.parent if best.exists() else save_dir
            self._log_queue.put(("line", f"\n✅ Treino concluído. Modelo salvo em: {best}"))
            self._log_queue.put(("done_ok", str(best)))
        except Exception as e:
            self._log_queue.put(("line", f"\n❌ Erro durante o treino: {e}"))
            self._log_queue.put(("done_err", str(e)))

    # ── Log / fila ──────────────────────────────────────────────────────
    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "done_ok":
                    self._on_training_done(ok=True)
                elif kind == "done_err":
                    self._on_training_done(ok=False)
                elif kind == "epoch":
                    self._on_epoch(*payload)
                else:
                    self._append_line(payload, progress=(kind == "progress"))
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    def _on_epoch(self, epoch, epochs, metrics, final):
        self.progress.configure(maximum=epochs, value=epoch)
        self.epoch_text.set("Validação final" if final else f"Época {epoch}/{epochs}")
        suffix = "(M)" if self.task.get() == "seg" else "(B)"   # máscara na segmentação, caixa na detecção
        for key, label in self._metric_vals.items():
            val = metrics.get(f"metrics/{key}{suffix}")
            if val is not None:
                label.config(text=f"{float(val):.3f}")

    def _on_training_done(self, ok):
        self.btn_train.config(state="normal", text="Iniciar treinamento")
        if ok:
            self.status_text.set("Treino concluído ✅")
            self.btn_open_folder.config(state="normal")
        else:
            self.status_text.set("Erro no treino ❌")

    def _append_line(self, text, progress=False):
        self.log_text.config(state="normal")
        if progress and self._last_line_was_progress:
            self.log_text.delete("end-2l", "end-1l")  # troca a ultima linha em vez de empilhar
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self._last_line_was_progress = progress

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._last_line_was_progress = False

    def _open_result_folder(self):
        if not self._last_save_dir:
            return
        path = str(self._last_save_dir)
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: on nao-Windows isso nem existe, so entra aqui no Windows
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])


if __name__ == "__main__":
    app = TrainerApp()
    app.mainloop()
