# Tech Challenge Phase 1 - Instruções de Execução

Este diretório contém scripts e notebooks para reproduzir o pipeline de visão em CPU: extração de features com ResNet18 pré-treinada, treino de classificador (RandomForest) e inferência.

Pré-requisitos
- Python 3.10+ recomendado (o ambiente do projeto usa venv em workspace/venv)
- Virtualenv ativado: `source /home/feijolo/.openclaw/workspace/venv/bin/activate`
- Instalar dependências: `pip install -r requirements.txt`

Fluxo de execução (recomendado, CPU)
1) Verifique os dados:
   - `tech_challenge_phase1/data/uid_label_map.csv` deve conter colunas: `uid,label,patient_id`
   - Imagens em `tech_challenge_phase1/data/cbis_ddsm_demo/` (uids correspondentes)

2) Extrair features (executar em background / CPU):
   ```
   python tech_challenge_phase1/scripts/extract_features.py --project-dir tech_challenge_phase1 --img-dir data/cbis_ddsm_demo --uid-csv data/uid_label_map.csv --out-dir features --img-size 128 --batch-size 32 --num-workers 4
   ```
   Saída: `tech_challenge_phase1/features/features.npz`

3) Treinar classificador:
   ```
   python tech_challenge_phase1/scripts/train_classifier.py --project-dir tech_challenge_phase1 --features-dir features --out-dir models --n-estimators 200
   ```
   Saída: `tech_challenge_phase1/models/classifier_rf.joblib` e relatórios em `tech_challenge_phase1/models/`

4) Inferência:
   ```
   python tech_challenge_phase1/scripts/inference.py --project-dir tech_challenge_phase1 --features-dir features --model models/classifier_rf.joblib --out reports/predictions.csv
   ```

5) Entrega:
   - Relatórios finais e figuras: `tech_challenge_phase1/models/` e `tech_challenge_phase1/reports/`
   - Zipar para submissão (sem venv): `zip -r submission.zip tech_challenge_phase1 -x "*/venv/*"`

Observações
- Para performance em CPU: ajustar `--img-size` para 128 e `--batch-size` conforme CPU/RAM.
- Para reprodutibilidade: defina `OMP_NUM_THREADS` e `torch.set_num_threads()` conforme CPU.

Se quiser, eu executo agora a extração de features localmente (vai demorar dependendo do número de imagens).