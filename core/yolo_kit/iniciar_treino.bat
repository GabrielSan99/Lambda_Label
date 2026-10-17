cd /d "%~dp0"
echo === Pasta: %CD%
if not exist .venv\Scripts\python.exe (
  echo === Criando ambiente .venv com o Python do sistema...
  python -m venv .venv || goto erro_venv
)
echo === Python do ambiente:
.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.version)" || goto erro_venv
echo === Instalando dependencias (pip -v, detalhado)...
.venv\Scripts\python.exe -m pip install -v -r requirements.txt || goto erro_pip
echo === Verificando GPU no PyTorch:
.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| build CUDA:', torch.version.cuda, '| GPU disponivel:', torch.cuda.is_available())"
.venv\Scripts\python.exe -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
if not errorlevel 1 goto gpu_ok
where nvidia-smi >nul 2>&1
if errorlevel 1 goto gpu_ok
echo === GPU NVIDIA detectada, mas o PyTorch esta sem CUDA. Instalando build CUDA (download grande, so na primeira vez)...
.venv\Scripts\python.exe -m pip install -v --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126 || goto erro_cuda
.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| GPU disponivel:', torch.cuda.is_available())"
:gpu_ok
echo === Iniciando treinador (python -u, sem buffer)...
.venv\Scripts\python.exe -u yolo_trainer_gui.py
echo === Treinador encerrado com codigo %ERRORLEVEL%
pause
goto fim
:erro_venv
echo.
echo FALHA ao criar o ambiente .venv. Confira se o Python 3 esta instalado
echo (python.org, marcando "Add python.exe to PATH") e tente de novo.
pause
goto fim
:erro_pip
echo.
echo FALHA ao instalar as dependencias (pip). Veja a mensagem de erro acima.
echo Causa comum: internet caiu no meio do download -- rode de novo.
pause
goto fim
:erro_cuda
echo.
echo FALHA ao instalar o PyTorch com CUDA. Veja a mensagem de erro acima.
pause
:fim
