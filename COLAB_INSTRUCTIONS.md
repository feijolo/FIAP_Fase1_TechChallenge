Colab — Fase B: Treino CNN (ResNet50) — instruções rápidas

Objetivo
- Treinar um classificador por transfer learning (ResNet50) em GPU (Colab) usando o conjunto de imagens já organizado em tech_challenge_phase1/data.
- Salvar modelo, mapas Grad-CAM e relatórios em tech_challenge_phase1/models e tech_challenge_phase1/reports/gradcam_colab.

Preparo (no Colab)
1) Abra https://colab.research.google.com/ e selecione "New Python 3 notebook".
2) Runtime > Change runtime type > GPU (preferência: Tesla T4/P100/V100).
3) Clone este repositório ou suba os arquivos do workspace:
   - Para clonar do Git (se estiver em um repo remoto):
     !git clone <SEU_REPO_URL>
     %cd tech_challenge_phase1
   - Ou faça upload do diretório tech_challenge_phase1 pelo painel esquerdo (Files > Upload).

Instalação de dependências (no Colab)
Execute no notebook:

!pip install -r requirements-colab.txt

Se usar Kaggle para baixar CBIS‑DDSM, faça upload do seu kaggle.json para /root/.kaggle/kaggle.json (ou use Colab file upload) e então rode os comandos de download (veja README_PROJECT_PLAN.md).

Executar treino (script preparado)
No notebook (após instalar dependências e garantir que os dados estejam em tech_challenge_phase1/data/cbis_ddsm_demo):

!python3 tech_challenge_phase1/scripts/train_cnn_colab.py \
  --data-dir tech_challenge_phase1/data/cbis_ddsm_demo \
  --uid-csv tech_challenge_phase1/data/uid_label_map.csv \
  --out-dir tech_challenge_phase1/models \
  --reports-dir tech_challenge_phase1/reports/gradcam_colab \
  --epochs 10 \
  --batch-size 32 \
  --img-size 224

O script salva:
- models/cnn_resnet50.pth (pesos)
- reports/gradcam_colab/ (exemplos Grad-CAM)
- reports/cnn_classification_report.json

Notas e recomendações
- Ajuste --epochs e --batch-size conforme a GPU disponível; para T4, batch 16–32 é razoável.
- Para reprodução: defina seed via --seed.
- Se preferir treinar em Colab Pro (mais VRAM), aumente batch-size e img-size.
- O script inclui checkpoints automáticos (melhor val acc).

Pronto — quer que eu crie também um notebook .ipynb pronto para Colab ou executo o script localmente aqui (vai usar CPU e pode demorar)?
