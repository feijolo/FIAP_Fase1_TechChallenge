# 🩺 Tech Challenge Fase 1 — Diagnóstico de Câncer de Mama com Machine Learning

> **Projeto acadêmico** desenvolvido para o Tech Challenge Fase 1 do curso de IA para Devs da FIAP.

---

## 📌 O que é este projeto?

Este projeto constrói um **sistema inteligente de apoio ao diagnóstico de câncer de mama**. Ele recebe dados numéricos extraídos de exames celulares (aspirado por agulha fina — FNA) e classifica cada amostra como **tumor maligno** ou **tumor benigno**, utilizando algoritmos de Machine Learning.

### Por que ele existe?

Uma rede de hospitais deseja implementar uma ferramenta de IA que auxilie médicos na triagem de pacientes com suspeita de câncer de mama, permitindo:

- **Identificação precoce** de casos potencialmente malignos
- **Priorização de atendimento** — casos suspeitos são encaminhados mais rapidamente
- **Apoio à decisão** — o modelo complementa (nunca substitui) o julgamento do médico

> ⚠️ **Importante**: Este sistema é uma **ferramenta de suporte**. O diagnóstico final deve ser sempre dado por um profissional de saúde.

---

## 📊 Dados utilizados

### Dados tabulares (obrigatório)

| Item | Detalhe |
|------|---------|
| **Dataset** | [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) |
| **Origem** | UCI Machine Learning Repository |
| **Amostras** | 569 (212 malignos, 357 benignos) |
| **Features** | 30 características numéricas (raio, textura, perímetro, área, suavidade etc.) |
| **Carregamento** | Automático via `sklearn.datasets` — não é necessário baixar nada manualmente |

### Imagens de mamografia (extra)

| Item | Detalhe |
|------|---------|
| **Dataset** | [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) |
| **Técnica** | Transfer Learning com ResNet18 (PyTorch) |
| **Interpretabilidade** | Grad-CAM — visualização das regiões mais relevantes na imagem |

---

## 🗂️ Estrutura do repositório

```
FIAP_Fase1_TechChallenge/
│
├── README.md                        ← Este arquivo (visão geral e instruções)
├── COLAB_INSTRUCTIONS.md            ← Instruções para treino CNN no Google Colab
├── Dockerfile                       ← Arquivo para execução via container Docker
├── requirements.txt                 ← Dependências Python (pipeline tabular + CNN)
├── .gitignore                       ← Arquivos ignorados pelo Git
│
├── src/                             ← Código-fonte reutilizável
│   ├── data_utils.py                   Funções para carregar e preparar os dados
│   └── modeling.py                     Funções auxiliares de modelagem
│
├── scripts/                         ← Scripts executáveis
│   ├── run_tabular_pipeline.py         Pipeline completo: EDA → Treino → Avaliação → SHAP
│   ├── inference.py                    Faz previsões usando um modelo já treinado
│   ├── extract_features.py            Extrai features de imagens com CNN pré-treinada
│   ├── train_classifier.py            Treina classificador sobre features extraídas
│   └── train_cnn_pytorch.py           Treino CNN end-to-end com PyTorch
│
├── notebooks/                       ← Notebooks Jupyter
│   ├── 0_setup_and_eda.ipynb           Análise exploratória completa dos dados tabulares
│   └── colab_train.ipynb               Treino da CNN no Google Colab (GPU)
│
├── models/                          ← Modelos treinados (gerados automaticamente)
│   ├── logreg.joblib                   Logistic Regression treinado
│   ├── rf.joblib                       Random Forest treinado
│   ├── scaler.joblib                   StandardScaler ajustado
│   └── classifier_rf.joblib            Classificador RF sobre features de imagem
│
├── reports/                         ← Resultados e visualizações (gerados automaticamente)
│   ├── tabular_summary.json            Resumo completo das métricas
│   ├── cm_*.png                        Matrizes de confusão
│   ├── roc_*.png                       Curvas ROC
│   ├── shap_*.png                      Análises SHAP (importância das features)
│   ├── correlation_heatmap.png         Heatmap de correlação entre features
│   └── gradcam/                        Visualizações Grad-CAM (CNN)
│
└── features/                        ← Features extraídas de imagens (CNN)
    └── features.npz                    Array NumPy com features e labels
```

---

## ⚙️ Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.10** ou superior — [Download](https://www.python.org/downloads/)
- **pip** — gerenciador de pacotes do Python (já incluído com Python 3.10+)
- **Git** — para clonar o repositório — [Download](https://git-scm.com/)
- *(Opcional)* **Docker** — apenas se quiser executar via container

---

## 🚀 Como executar

Escolha **uma** das três opções abaixo.

### Opção 1 — Execução local (recomendada para iniciantes)

Siga os passos abaixo no terminal (Linux/Mac) ou no Prompt de Comando/PowerShell (Windows):

```bash
# 1. Clone o repositório para sua máquina
git clone https://github.com/feijolo/FIAP_Fase1_TechChallenge.git

# 2. Entre na pasta do projeto
cd FIAP_Fase1_TechChallenge

# 3. Crie um ambiente virtual Python (isola as dependências do projeto)
python -m venv venv

# 4. Ative o ambiente virtual
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows (PowerShell)
# venv\Scripts\activate.bat     # Windows (CMD)

# 5. Instale as dependências do projeto
pip install -r requirements.txt

# 6. Execute o pipeline tabular completo
python scripts/run_tabular_pipeline.py
```

**O que acontece ao executar o passo 6?**
1. O dataset é carregado automaticamente (não precisa baixar nada)
2. Uma análise exploratória é realizada e gráficos são salvos em `reports/`
3. Dois modelos são treinados (Logistic Regression e Random Forest)
4. Métricas de avaliação são calculadas e salvas
5. Análises SHAP de interpretabilidade são geradas
6. Modelos treinados são salvos em `models/`

### Opção 2 — Execução via Docker

```bash
# 1. Construa a imagem Docker
docker build -t tech-challenge-fase1 .

# 2. Execute o pipeline (resultados e modelos são salvos na sua máquina)
docker run -v $(pwd)/reports:/app/reports -v $(pwd)/models:/app/models tech-challenge-fase1
```

> A flag `-v` mapeia as pastas `reports/` e `models/` do container para sua máquina local, para que você possa acessar os resultados depois.

### Opção 3 — Google Colab (apenas para CNN com GPU)

Para treinar a rede neural convolucional (CNN) com aceleração por GPU:

1. Abra o notebook [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) no Google Colab
2. Siga as instruções detalhadas no arquivo [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md)

---

## 🔬 Metodologia passo a passo

O pipeline segue 5 etapas, executadas automaticamente pelo script `run_tabular_pipeline.py`:

### Etapa 1 — Análise Exploratória (EDA)

**Objetivo**: Entender os dados antes de treinar qualquer modelo.

- Estatísticas descritivas (média, desvio padrão, mínimo, máximo)
- Distribuição das classes: 63% benigno vs 37% maligno
- Histogramas e box plots comparando características de tumores malignos vs benignos
- Heatmap de correlação para identificar features redundantes (multicolinearidade)

📓 Detalhes completos no notebook [`notebooks/0_setup_and_eda.ipynb`](notebooks/0_setup_and_eda.ipynb)

### Etapa 2 — Pré-processamento

**Objetivo**: Preparar os dados para que os modelos aprendam de forma eficiente.

- Verificação de valores ausentes (nenhum encontrado neste dataset)
- **Normalização** com `StandardScaler` — coloca todas as features na mesma escala (essencial para Logistic Regression)
- Divisão dos dados em **3 conjuntos** com estratificação (mesma proporção de classes em cada):
  - **Treino** (60%) — usado para treinar os modelos
  - **Validação** (20%) — usado para ajustar hiperparâmetros
  - **Teste** (20%) — usado apenas na avaliação final (nunca visto pelo modelo durante o treino)

### Etapa 3 — Modelagem

**Objetivo**: Treinar modelos que aprendam a distinguir tumores malignos de benignos.

| Modelo | O que faz | Por que foi escolhido |
|--------|-----------|----------------------|
| **Logistic Regression** | Traça uma fronteira linear entre as classes | Simples, interpretável, serve como baseline. A regularização L2 evita overfitting mesmo com features correlacionadas |
| **Random Forest** | Combina decisões de múltiplas árvores | Captura padrões não-lineares, é robusto a multicolinearidade e fornece ranking de importância das features |

### Etapa 4 — Avaliação

**Objetivo**: Medir a qualidade dos modelos com dados que eles nunca viram.

- **Cross-validation** estratificada (5 folds) no conjunto treino+validação
- Métricas calculadas:
  - **Accuracy** — % de acertos geral
  - **Recall** (sensibilidade) — % de tumores malignos corretamente detectados
  - **F1-Score** — média harmônica entre precisão e recall
  - **ROC-AUC** — capacidade do modelo de separar as duas classes
- Visualizações geradas: matrizes de confusão e curvas ROC/Precision-Recall

### Etapa 5 — Interpretabilidade

**Objetivo**: Explicar *por que* o modelo toma cada decisão.

- **Feature Importance** (Random Forest) — quais características mais influenciam a decisão
- **SHAP Values** (ambos os modelos) — mostra o impacto de cada feature em cada previsão individual

---

## 📈 Resultados

### Métricas no conjunto de teste

| Modelo | Accuracy | Recall (Maligno) | F1 (Maligno) | ROC-AUC |
|--------|----------|-------------------|--------------|---------|
| Logistic Regression | ~97.4% | ~95.3% | ~96.5% | ~0.997 |
| Random Forest | ~96.5% | ~93.0% | ~95.2% | ~0.995 |

> 💡 Os valores exatos podem variar ligeiramente entre execuções. Para gerar resultados atualizados, execute `python scripts/run_tabular_pipeline.py`.

### Por que priorizamos o Recall?

No diagnóstico de câncer de mama, os erros têm pesos diferentes:

| Tipo de erro | Consequência | Gravidade |
|--------------|-------------|-----------|
| **Falso Negativo** (tumor maligno classificado como benigno) | Paciente pode não receber tratamento a tempo | 🔴 **Crítica** — pode ser fatal |
| **Falso Positivo** (tumor benigno classificado como maligno) | Paciente faz exames complementares desnecessários | 🟡 **Moderada** — gera custos, mas sem risco de vida |

Por isso, o **Recall** (taxa de detecção de malignos) é a métrica mais importante neste contexto.

### Limitações conhecidas

1. **Dataset pequeno** (569 amostras) — insuficiente para uso clínico real
2. **Fonte única** (UCI/Wisconsin) — pode não representar populações diversas
3. **Features pré-extraídas** — o modelo não processa imagens diretamente
4. **Sem validação prospectiva** — não foi testado em ambiente clínico real

---

## 📂 O que cada script faz

| Script | Comando | Descrição |
|--------|---------|-----------|
| `run_tabular_pipeline.py` | `python scripts/run_tabular_pipeline.py` | Executa todo o pipeline: carrega dados, faz EDA, treina modelos, avalia e gera SHAP |
| `inference.py` | `python scripts/inference.py --model models/classifier_rf.joblib` | Faz previsões usando um modelo já salvo sobre features de imagem |
| `extract_features.py` | `python scripts/extract_features.py` | Extrai features de imagens usando CNN pré-treinada |
| `train_classifier.py` | `python scripts/train_classifier.py` | Treina um classificador Random Forest sobre features extraídas |
| `train_cnn_pytorch.py` | `python scripts/train_cnn_pytorch.py` | Treina CNN ResNet18 end-to-end com PyTorch |

---

## ✅ Entregáveis do projeto

- [x] Código-fonte completo em Python
- [x] README.md com instruções claras de execução
- [x] Dockerfile para execução containerizada
- [x] Dataset carregado automaticamente via `sklearn` (ou [download Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data))
- [x] Resultados: gráficos, matrizes de confusão, curvas ROC, SHAP
- [x] Relatório técnico (este README + `reports/tabular_summary.json`)
- [x] Notebook de Análise Exploratória (`notebooks/0_setup_and_eda.ipynb`)
- [x] **EXTRA**: CNN com Transfer Learning (ResNet18) + Grad-CAM

---

## 📚 Referências

1. **Breast Cancer Wisconsin Dataset** — W.H. Wolberg, W.N. Street, O.L. Mangasarian (1995). UCI Machine Learning Repository.
   https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

2. **CBIS-DDSM** — Curated Breast Imaging Subset of DDSM.
   https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

3. **SHAP** — Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.

4. **scikit-learn** — Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
   https://scikit-learn.org/

---

## 👥 Autores

Projeto desenvolvido por alunos do curso **IA para Devs — FIAP** como parte do Tech Challenge Fase 1.