Plano de Ação - Tech Challenge Fase 1

Resumo do que vou entregar e cronograma (fases):

Fase A - Tabular (obrigatório) — 3 dias
- Baixar e preparar dataset Breast Cancer Wisconsin (sklearn/Kaggle)
- EDA, estatísticas descritivas e visualizações
- Pré-processamento: tratamento de valores ausentes, encoding, scaling
- Modelagem: Logistic Regression, RandomForest (+KNN se tempo)
- Treino/validação/teste com cross-validation
- Métricas: accuracy, precision, recall, F1, ROC-AUC
- Interpretabilidade: feature importance e SHAP
- Notebook executável, scripts e relatório em PDF

Fase B - Visão Computacional (extra) — 5 dias
- Baixar subset de mamografias (CBIS-DDSM) ou Chest X-Ray Pneumonia via Kaggle
- Preparar dataset com pipeline de augmentations
- Treinar CNN por transfer learning (ResNet50) em Colab/TPU/GPU sugerido
- Avaliação (confusion matrix, precision/recall/F1, ROC) e interpretação (Grad-CAM)
- Notebook Colab com instruções e scripts

Fase C - Documentação e entrega — 1 dia
- Gerar relatório final em PDF com resumo, metodologia, resultados e limitações
- Organizar repositório pronto para submissão

Próximos passos imediatos (com sua autorização):
- Instalar dependências base no ambiente
- Executar pipeline tabular end-to-end e gerar resultados iniciais

Observações:
- Datasets grandes serão baixados via Kaggle API; preciso que você confirme se quer que eu use sua conta (coloque ~/.kaggle/kaggle.json) ou que eu use datasets menores/localizados por sklearn.
- Treinamento de CNN full será feito preferencialmente em Colab (recomendo GPU disponível). Vou preparar tudo para rodar lá com instruções.

Se estiver OK com o plano, responda "OK prosseguir" e eu começo pela Fase A (tabular).