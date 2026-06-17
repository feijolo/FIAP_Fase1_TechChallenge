# ☁️ Instruções para treino da CNN no Google Colab

> Este guia explica como treinar a rede neural convolucional (CNN) com GPU gratuita do Google Colab.

---

## 🎯 Objetivo

Treinar um classificador de imagens de mamografia usando **Transfer Learning** (ResNet18 pré-treinada) para distinguir lesões benignas de malignas, com visualização de interpretabilidade via **Grad-CAM**.

---

## 📋 Pré-requisitos

- Uma conta Google (para acessar o Google Colab)
- *(Opcional)* Uma conta Kaggle com API key (para download automático do dataset CBIS-DDSM)

---

## 🚀 Passo a passo

### Passo 1 — Abrir o Google Colab

1. Acesse [https://colab.research.google.com/](https://colab.research.google.com/)
2. Clique em **"Novo notebook"** (ou abra diretamente o arquivo `notebooks/colab_train.ipynb` do repositório)

### Passo 2 — Ativar a GPU

1. No menu superior, clique em **Runtime** → **Change runtime type**
2. Em **Hardware accelerator**, selecione **GPU** (recomendado: T4, P100 ou V100)
3. Clique em **Save**

> 💡 Sem GPU o treino funciona, mas será muito mais lento.

### Passo 3 — Clonar o repositório

Execute esta célula no notebook:

```python
!git clone https://github.com/feijolo/FIAP_Fase1_TechChallenge.git
%cd FIAP_Fase1_TechChallenge
```

> **Alternativa**: Faça upload manual dos arquivos do projeto pelo painel lateral esquerdo (ícone de pasta → Upload).

### Passo 4 — Instalar dependências

```python
!pip install -r requirements.txt
```

### Passo 5 — Preparar o dataset de imagens

**Opção A — Download via Kaggle API** (recomendado para o dataset completo):

1. Acesse [https://www.kaggle.com/settings](https://www.kaggle.com/settings) e clique em **"Create New Token"** para baixar seu `kaggle.json`
2. No Colab, faça upload do arquivo:
   ```python
   from google.colab import files
   files.upload()  # Selecione o arquivo kaggle.json
   ```
3. Mova para o local correto:
   ```python
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```
4. Baixe o dataset:
   ```python
   !kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset
   !unzip -q cbis-ddsm-breast-cancer-image-dataset.zip -d data/cbis_ddsm_demo
   ```

**Opção B — Upload manual**: Suba as imagens já organizadas para a pasta `data/cbis_ddsm_demo/` com subpastas `train/` e `val/`, cada uma contendo subpastas por classe (`benign/`, `malignant/`).

### Passo 6 — Executar o treino

```python
!python scripts/train_cnn_pytorch.py
```

Ou, para controle fino dos parâmetros:

```python
!python scripts/train_cnn_pytorch.py \
  --data-dir data/cbis_ddsm_demo \
  --out-dir models \
  --reports-dir reports/gradcam \
  --epochs 10 \
  --batch-size 32 \
  --img-size 224
```

### Passo 7 — Verificar os resultados

Após o treino, os seguintes arquivos são gerados:

| Arquivo | Descrição |
|---------|-----------|
| `models/cnn_resnet50.pth` | Pesos do modelo treinado |
| `reports/cnn_classification_report.json` | Métricas de avaliação (precision, recall, F1) |
| `reports/gradcam/` | Imagens Grad-CAM mostrando as regiões de atenção do modelo |

Para baixar os resultados para sua máquina:

```python
from google.colab import files
files.download('reports/cnn_classification_report.json')
# Ou compacte tudo:
!zip -r resultados.zip models/ reports/
files.download('resultados.zip')
```

---

## ⚙️ Ajustes e dicas

| Parâmetro | Recomendação |
|-----------|-------------|
| `--epochs` | 10 para teste rápido, 30+ para melhores resultados |
| `--batch-size` | 16–32 para GPU T4; aumente se usar Colab Pro (mais VRAM) |
| `--img-size` | 224 (padrão ResNet); aumente para 384 se houver memória disponível |
| `--seed` | Defina um valor fixo (ex: `--seed 42`) para reprodutibilidade |

> O script inclui **checkpoints automáticos** — ele salva o modelo com melhor acurácia na validação, mesmo que o treino seja interrompido.
