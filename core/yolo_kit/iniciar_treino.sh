#!/usr/bin/env bash
set -x
cd "$(dirname "$0")" || exit 1
echo "=== Pasta: $(pwd)"
if [ ! -x .venv/bin/python ]; then
  echo "=== Criando ambiente .venv com o Python do sistema..."
  python3 -m venv .venv || { echo "Instale o Python 3 (com venv) e tente de novo."; read -rp "Enter pra fechar..."; exit 1; }
fi
echo "=== Python do ambiente:"
.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)"
echo "=== Instalando dependencias (pip -v, detalhado)..."
.venv/bin/python -m pip install -v -r requirements.txt || { echo "Falha ao instalar as dependencias."; read -rp "Enter pra fechar..."; exit 1; }
echo "=== Verificando GPU no PyTorch:"
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| build CUDA:', torch.version.cuda, '| GPU disponivel:', torch.cuda.is_available())"
if ! .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU NVIDIA detectada, mas o PyTorch esta sem CUDA. Instalando build CUDA (download grande, so na primeira vez)..."
  .venv/bin/python -m pip install -v --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126 || { echo "Falha ao instalar o PyTorch com CUDA."; read -rp "Enter pra fechar..."; exit 1; }
  .venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPU disponivel:', torch.cuda.is_available())"
fi
echo "=== Iniciando treinador (python -u, sem buffer)..."
.venv/bin/python -u yolo_trainer_gui.py
echo "=== Treinador encerrado com codigo $?"
read -rp "Enter pra fechar..."
