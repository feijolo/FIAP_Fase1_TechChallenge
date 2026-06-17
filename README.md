# Tech Challenge Fase 1 — Diagnóstico de Câncer de Mama com Machine Learning

## Descrição do Projeto

Sistema inteligente de suporte ao diagnóstico de câncer de mama, desenvolvido como parte do Tech Challenge Fase 1 da FIAP. O projeto utiliza algoritmos de Machine Learning para classificar tumores como **malignos** ou **benignos** a partir de características celulares extraídas de exames de aspirado por agulha fina (FNA).

### Problema

Uma rede de hospitais busca implementar um sistema de IA capaz de auxiliar profissionais de saúde na identificação precoce de câncer de mama, acelerando a triagem e apoiando decisões médicas.

### Dataset

- **Breast Cancer Wisconsin (Diagnostic)** — [Kaggle/UCI](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- 569 amostras, 30 features numéricas
- Classes: Maligno (212) e Benigno (357)
- O dataset é carregado automaticamente via `sklearn.datasets`

### EXTRA — Visão Computacional (CNN)

- Detecção de câncer de mama em mamografias usando [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- Transfer learning com ResNet18
- Interpretabilidade com Grad-CAM

---

## Estrutura do Repositório

```
├── Dockerfile                  # Container para execução do pipeline
├── README.md                   # Este arquivo
├── requirements.txt            # Dependências Python
├── src/
│   ├── data_utils.py           # Utilitários de carregamento de dados
│   └── modeling.py             # Funções de modelagem
├── scripts/
│   ├── run_tabular_pipeline.py # Pipeline principal (EDA + Modelos + SHAP)
│   ├── extract_features.py     # Extração de features de imagens (CNN)
│   ├── train_classifier.py     # Treino de classificador sobre features
│   ├── train_cnn_pytorch.py    # Treino CNN end-to-end (PyTorch)
│   └── inference.py            # Inferência com modelo salvo
├── notebooks/
│   ├── 0_setup_and_eda.ipynb   # Análise exploratória completa
│   └── colab_train.ipynb       # Treino CNN no Google Colab
├── reports/                    # Resultados, gráficos e relatórios
│   ├── tabular_summary.json    # Resumo completo dos resultados
│   ├── cm_*.png                # Matrizes de confusão
│   ├── roc_*.png               # Curvas ROC
│   ├── shap_*.png              # Análises SHAP
│   ├── correlation_heatmap.png # Heatmap de correlação
│   └── gradcam/                # Visualizações Grad-CAM (CNN)
├── models/                     # Modelos treinados
│   ├── logreg.joblib           # Logistic Regression
│   ├── rf.joblib               # Random Forest
│   └── scaler.joblib           # StandardScaler
└── features/                   # Features extraídas (CNN)
```

---

## Pré-requisitos

- Python 3.10+
- pip (gerenciador de pacotes)

## Instalação e Execução

### Opção 1: Execução Local

```bash
# Clonar o repositório
git clone https://github.com/feijolo/FIAP_Fase1_TechChallenge.git
cd FIAP_Fase1_TechChallenge

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar o pipeline tabular completo
python scripts/run_tabular_pipeline.py
```

### Opção 2: Docker

```bash
# Construir a imagem
docker build -t tech-challenge-fase1 .

# Executar o pipeline
docker run -v $(pwd)/reports:/app/reports -v $(pwd)/models:/app/models tech-challenge-fase1
```

### Opção 3: Google Colab (CNN)

Para treinar a CNN com GPU, abra o notebook `notebooks/colab_train.ipynb` no Google Colab. Veja instruções detalhadas em `COLAB_INSTRUCTIONS.md`.

---

## Pipeline e Metodologia

### 1. Análise Exploratória (EDA)

Detalhada no notebook `notebooks/0_setup_and_eda.ipynb`:
- Estatísticas descritivas e inspeção de qualidade dos dados
- Distribuição das classes (63% benigno vs 37% maligno)
- Histogramas e box plots por classe
- **Análise de correlação** com heatmap e identificação de multicolinearidade
- Identificação de padrões celulares discriminativos

### 2. Pré-processamento

- Verificação de valores ausentes (nenhum encontrado)
- **StandardScaler** para normalização das features (necessário para Logistic Regression)
- Separação **Train/Validation/Test** (60/20/20) com estratificação

### 3. Modelagem

Dois modelos de classificação foram treinados e comparados:

| Modelo | Justificativa |
|--------|--------------|
| **Logistic Regression** | Modelo linear interpretável, serve como baseline. Com regularização L2, é robusto mesmo com features correlacionadas. |
| **Random Forest** | Modelo ensemble baseado em árvores, captura relações não-lineares e é robusto a multicolinearidade. Fornece feature importance nativa. |

### 4. Avaliação

- **Cross-validation** estratificada (5-fold) no conjunto train+val
- Métricas: **Accuracy, Recall, F1-Score, ROC-AUC**
- Matrizes de confusão e curvas ROC/Precision-Recall

### 5. Interpretabilidade

- **Feature Importance** (Random Forest)
- **SHAP Values** (ambos os modelos) — explicabilidade global e local das previsões

---

## Resultados

### Métricas no Conjunto de Teste

| Modelo | Accuracy | Recall (Maligno) | F1 (Maligno) | ROC-AUC |
|--------|----------|-------------------|--------------|---------|
| Logistic Regression | ~97.4% | ~95.3% | ~96.5% | ~0.997 |
| Random Forest | ~96.5% | ~93.0% | ~95.2% | ~0.995 |

> **Nota**: Os valores exatos podem variar ligeiramente entre execuções. Execute `python scripts/run_tabular_pipeline.py` para gerar resultados atualizados.

### Escolha de Métricas

Em diagnóstico de câncer de mama, a métrica prioritária é o **Recall** (sensibilidade) para a classe maligna:
- **Falsos negativos** (não detectar um tumor maligno) podem atrasar o tratamento e ser fatais
- **Falsos positivos** (classificar benigno como maligno) geram exames adicionais, mas não põem a vida em risco
- O **F1-score** complementa o recall ao equilibrar com precisão
- O **ROC-AUC** avalia a capacidade discriminativa geral do modelo

### Discussão Crítica

O modelo apresenta resultados promissores, mas existem limitações importantes:

1. **Dataset pequeno** (569 amostras) — insuficiente para generalização clínica robusta
2. **Fonte única** (UCI/Wisconsin) — pode não representar populações diversas
3. **Features pré-extraídas** — não cobre todo o fluxo diagnóstico real
4. **Sem validação prospectiva** em ambiente clínico

**Recomendação de uso**: O modelo deve ser utilizado **exclusivamente como ferramenta de suporte** à decisão médica, jamais como diagnóstico definitivo. O médico sempre deve ter a palavra final. Sugere-se uso como sistema de triagem para priorizar casos suspeitos.

---

## Entregáveis

- [x] Código-fonte completo em Python
- [x] README.md com instruções de execução
- [x] Dockerfile para execução containerizada
- [x] Dataset carregado automaticamente via sklearn (ou [link Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data))
- [x] Resultados: gráficos, matrizes de confusão, curvas ROC, SHAP
- [x] Relatório técnico (este README + `reports/tabular_summary.json`)
- [x] Notebook EDA (`notebooks/0_setup_and_eda.ipynb`)
- [x] EXTRA: CNN com transfer learning + Grad-CAM

---

## Referências

- Breast Cancer Wisconsin Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- CBIS-DDSM: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- SHAP: Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
- scikit-learn: https://scikit-learn.org/