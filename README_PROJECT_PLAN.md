# 📋 Plano de Execução do Projeto — Tech Challenge Fase 1

> Este documento descreve o plano de trabalho seguido para desenvolver o projeto de diagnóstico de câncer de mama com Machine Learning.

---

## 🎯 Objetivo geral

Construir um sistema de Machine Learning capaz de classificar tumores de mama como **malignos** ou **benignos**, utilizando dados clínicos tabulares (obrigatório) e imagens de mamografia (extra).

---

## 📅 Fases do projeto

### Fase A — Pipeline Tabular (obrigatório)

**Duração estimada**: 3 dias

**O que foi feito:**

1. **Carregamento dos dados**
   - Dataset Breast Cancer Wisconsin carregado automaticamente via `sklearn.datasets`
   - Sem necessidade de download manual ou conta Kaggle

2. **Análise Exploratória (EDA)**
   - Estatísticas descritivas (média, mediana, desvio padrão)
   - Visualizações: histogramas, box plots, heatmap de correlação
   - Identificação da distribuição das classes (63% benigno vs 37% maligno)

3. **Pré-processamento**
   - Verificação de valores ausentes
   - Normalização com `StandardScaler`
   - Divisão treino (60%) / validação (20%) / teste (20%) com estratificação

4. **Modelagem e treinamento**
   - Logistic Regression (baseline interpretável)
   - Random Forest (captura relações não-lineares)
   - Cross-validation estratificada (5-fold)

5. **Avaliação**
   - Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Matrizes de confusão e curvas ROC/Precision-Recall

6. **Interpretabilidade**
   - Feature Importance (Random Forest)
   - SHAP Values (ambos os modelos)

**Artefatos gerados:**
- Notebook EDA: `notebooks/0_setup_and_eda.ipynb`
- Script do pipeline: `scripts/run_tabular_pipeline.py`
- Modelos salvos: `models/logreg.joblib`, `models/rf.joblib`, `models/scaler.joblib`
- Relatórios: `reports/tabular_summary.json`, gráficos em `reports/`

---

### Fase B — Visão Computacional / CNN (extra)

**Duração estimada**: 5 dias

**O que foi feito:**

1. **Preparação do dataset de imagens**
   - Subset do CBIS-DDSM (mamografias) baixado via Kaggle API
   - Pipeline de data augmentation (flip horizontal, normalização)

2. **Treino com Transfer Learning**
   - Arquitetura: ResNet18 pré-treinada no ImageNet
   - Fine-tuning: camada final substituída para classificação binária
   - Treinamento executado preferencialmente em Google Colab com GPU

3. **Avaliação**
   - Matriz de confusão, Precision/Recall/F1, ROC-AUC

4. **Interpretabilidade**
   - Grad-CAM: mapas de calor mostrando quais regiões da imagem influenciam a decisão

**Artefatos gerados:**
- Script de treino: `scripts/train_cnn_pytorch.py`
- Notebook Colab: `notebooks/colab_train.ipynb`
- Instruções Colab: `COLAB_INSTRUCTIONS.md`
- Relatórios CNN: `reports/cnn_classification_report.json`, `reports/gradcam/`

---

### Fase C — Documentação e Entrega

**Duração estimada**: 1 dia

**O que foi feito:**

1. Redação do README principal com instruções claras de execução
2. Organização da estrutura do repositório
3. Criação do Dockerfile para execução containerizada
4. Revisão final de todos os artefatos

---

## 🔧 Ferramentas e tecnologias utilizadas

| Categoria | Ferramenta |
|-----------|-----------|
| Linguagem | Python 3.10+ |
| ML tabulares | scikit-learn, SHAP |
| Deep Learning | PyTorch, torchvision, timm |
| Visualização | matplotlib, seaborn |
| Dados | pandas, numpy |
| Containerização | Docker |
| Ambiente GPU | Google Colab |