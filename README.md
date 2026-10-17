# Lambda Label

Anotação manual e automática + machine learning sobre imagens espectrais **já coletadas
por outros equipamentos**. É o módulo de *Annotations* e *Machine Learning* do SpectralCrops,
separado do hardware: em vez de capturar, o projeto **importa** os arquivos de cada
equipamento e trabalha com qualquer número de bandas.

## Rodando

```powershell
py -3.12 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python manage.py migrate        # cria o usuário admin / lambda123
venv\Scripts\python manage.py runserver 0.0.0.0:8000
```

Abra http://127.0.0.1:8000, entre com `admin` / `lambda123` (troque a senha em `/admin/`).
Testes: `venv\Scripts\python manage.py test core`.

## Fluxo

1. **Projetos** → criar projeto.
2. **⬆ Importar imagens** → enviar arquivos/pasta pelo navegador, ou indicar uma pasta do
   servidor. O app varre, reconhece o formato e mostra o que achou; você confere e importa.
   - **Vincular** (pasta do servidor): cubos ENVI e `.npy` são lidos direto do original via
     memmap — nada é copiado (um `.bil` do Resonon tem ~1 GB).
   - **Copiar**: tudo vai para `data/samples/`.
3. **Editor** (clique na amostra): box, elipse, polígono e ponto; Otsu, SAM espectral,
   box→polígono e 🤖 IA (YOLO / classificador espectral). Painel direito: espectro médio ±
   desvio da forma selecionada, visualizador de todas as bandas (pode virar o fundo do
   editor) e escolha das bandas R/G/B da composição colorida. **Salvar (Ctrl+S)** mede os ROIs.
4. **Análise**: espectro médio por grupo (rótulo da imagem ou label do ROI) e tabela por ROI.
5. **Machine Learning**: classificador espectral (Random Forest, SVM, Regressão Logística,
   KNN) treinado no próprio app, com validação cruzada **por imagem** quando possível; e os
   modelos YOLO (exporta dataset → treina no PC com a ferramenta → envia o `best.pt`).
6. **Exportar**: YOLO seg/det, COCO JSON, CSV dos espectros por ROI, e "Amostras + cubos"
   (mesmo `anotacoes.json` do SpectralCrops, com o cubo `.npy` de cada imagem).

## Formatos de importação

| Equipamento / formato | Arquivos | Observações |
|---|---|---|
| VideometerLab | `.hips` | Reflectância (%) e fluorescência; comprimentos de onda e mm/pixel vêm do arquivo. Convertido uma vez para `cube.npy`. |
| Hiperespectral ENVI (Resonon/Spectronon, Specim, Headwall…) | `.hdr` + `.bil`/`.bip`/`.bsq`/`.raw`/`.img` | Lido via memmap. Usa `reflectance scale factor`, `default bands`, `byte order`, `header offset`. |
| SpectralCrops | pasta com `raw/` + `calibrated/` (`365nm.png`…) ou pasta do export de amostras | Prefere `calibrated/`; usa o `rgb.png` como preview. Vale para qualquer equipamento que salve `NNNnm.png`. |
| Imagens | PNG, JPG, TIFF (multipágina, 16 bits), BMP | `A1_RGB.jpg` + `A1_CHL.jpg` + `A1_ARIIDX.jpg` (ex.: PhenoVation SeedReporter) viram **uma** amostra; sufixo `660nm` vira comprimento de onda. |
| **Qualquer outro equipamento** | `.npy` + `.json` opcional | Ver abaixo. |

### Integrando um equipamento novo

O caminho mais curto é salvar do lado do equipamento um cubo NumPy e (opcionalmente) um JSON
com o mesmo nome:

```python
import json, numpy as np
np.save("amostra01.npy", cubo)                 # (altura, largura, bandas)
json.dump({
    "wavelengths": [450.0, 550.0, 650.0],     # nm, um por banda (opcional)
    "band_names": ["azul", "verde", "vermelho"],
    "value_scale": 0.01,                       # valor * escala = fração 0..1 (padrão: pelo tipo do dado)
    "unit": "reflectance",                     # ou "intensity"
    "mm_per_pixel": 0.05,
    "layout": "HWB"                            # ou "BHW" se as bandas vierem primeiro
}, open("amostra01.json", "w"))
```

Importação automática (ex.: script que roda depois de cada coleta):

```powershell
venv\Scripts\python manage.py import_samples "Meu projeto" D:\coletas\2026-10-17 --link
```

Para um formato nativo, crie uma subclasse de `Importer` em `core/importers/` (métodos
`discover` e `load`) e registre em `core/importers/__init__.py`.

## Como os dados são guardados

- `data/samples/<projeto>/<id>/`: `preview.png` (fundo do editor), `thumb.jpg`, `meta.json`
  (metadados completos, legível fora do app), `cube.npy` (quando convertido) e `source/`
  (cópia do original no modo copiar).
- Valor armazenado × `value_scale` = fração 0..1. Espectros e ROIs são exibidos em %.
- Amostras de equipamentos diferentes no mesmo projeto: Análise e ML usam a grade de bandas
  mais comum do projeto e interpolam as outras quando cobrem a mesma faixa (as que não cobrem
  ficam de fora, e a tela diz quantas).

## Configuração

| Variável | Padrão | |
|---|---|---|
| `LAMBDA_DATA_DIR` | `./data` | onde ficam amostras, modelos e uploads temporários |
| `LAMBDA_ALLOW_LOCAL_IMPORT` | `1` | importar de pasta do servidor (só usuários staff) — desligue se expuser o app na rede |
| `LAMBDA_SECRET_KEY`, `LAMBDA_DEBUG` | dev | troque em produção |

YOLO no editor exige `pip install ultralytics` (comentado no `requirements.txt` por ser pesado).
