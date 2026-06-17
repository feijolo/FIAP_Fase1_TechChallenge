# 📓 Notebooks

Esta pasta contém os notebooks Jupyter do projeto. Eles podem ser abertos localmente com Jupyter ou diretamente no Google Colab.

---

## Notebooks disponíveis

### `0_setup_and_eda.ipynb` — Análise Exploratória de Dados (EDA)

**O que faz**: Carrega o dataset Breast Cancer Wisconsin e realiza uma análise exploratória completa dos dados tabulares.

**Conteúdo**:
- Carregamento e inspeção do dataset (shape, tipos, valores ausentes)
- Estatísticas descritivas por classe (maligno vs benigno)
- Histogramas e box plots das features
- Heatmap de correlação entre features
- Identificação de padrões celulares relevantes para classificação

**Como executar**:
```bash
# Na raiz do projeto, com o ambiente virtual ativo:
pip install -r requirements.txt
jupyter notebook notebooks/0_setup_and_eda.ipynb
```

---

### `colab_train.ipynb` — Treino da CNN no Google Colab

**O que faz**: Treina uma rede neural convolucional (ResNet18) para classificação de mamografias, usando GPU do Google Colab.

**Conteúdo**:
- Instalação de dependências no ambiente Colab
- Download e preparação do dataset de imagens (CBIS-DDSM)
- Treino com Transfer Learning (ResNet18 pré-treinada)
- Avaliação do modelo (métricas e matriz de confusão)
- Visualização Grad-CAM (interpretabilidade)

**Como executar**:
1. Abra o arquivo no Google Colab: clique em `Open in Colab` ou faça upload manual
2. Ative GPU: `Runtime > Change runtime type > GPU`
3. Siga as células do notebook em ordem

> 📌 Para instruções detalhadas, consulte o arquivo [`COLAB_INSTRUCTIONS.md`](../COLAB_INSTRUCTIONS.md) na raiz do projeto.