import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

DATA_DIR = Path('tech_challenge_phase1/data/cbis_ddsm_demo')
REPORT_DIR = Path('tech_challenge_phase1/reports')
MODELS_DIR = Path('tech_challenge_phase1/models')
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not DATA_DIR.exists():
    raise SystemExit(f'data dir not found: {DATA_DIR}')

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = ImageFolder(DATA_DIR, transform=transform_train)
# split train/val 80/20
n = len(train_dataset)
n_train = int(0.8*n)
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [n_train, n-n_train], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# replace final layer
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.unsqueeze(1).float().to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        correct += (preds.cpu().numpy() == labels.cpu().numpy().astype(int)).sum()
        total += imgs.size(0)
    train_loss = running_loss/total
    train_acc = correct/total

    # val
    model.eval()
    vloss = 0.0
    vcorrect = 0
    vtotal = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.unsqueeze(1).float().to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            vloss += loss.item()*imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            vcorrect += (preds.cpu().numpy() == labels.cpu().numpy().astype(int)).sum()
            vtotal += imgs.size(0)
            y_true.extend(labels.cpu().numpy().astype(int).ravel().tolist())
            y_pred.extend(preds.cpu().numpy().ravel().tolist())
    val_loss = vloss/vtotal
    val_acc = vcorrect/vtotal
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    print(f'Epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')

# save model
torch.save(model.state_dict(), MODELS_DIR/'cnn_resnet18_cpu.pth')
with open(REPORT_DIR/'cnn_history_pytorch.json','w') as f:
    json.dump({k:[float(x) for x in v] for k,v in history.items()}, f)

# classification report
report = classification_report(y_true, y_pred, target_names=list(train_dataset.classes), output_dict=True)
cm = confusion_matrix(y_true, y_pred)

pd.DataFrame(report).to_json(REPORT_DIR/'cnn_classification_report.json', orient='columns')
pd.DataFrame(cm).to_csv(REPORT_DIR/'cnn_confusion_matrix.csv', index=False)

print('Training finished. Model and reports saved.')
