#!/usr/bin/env python3
"""
Extract image features using a pretrained backbone (timm) and save to a compressed .npz
Usage:
  python scripts/extract_features.py --project-dir /path/to/tech_challenge_phase1 --img-dir data/cbis_ddsm_demo --uid-csv data/uid_label_map.csv --out-dir features --img-size 128 --batch-size 32
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, uid_col='uid', ext_guess=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.uid_col = uid_col
        self.ext_guess = ext_guess

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, uid):
        p = self.img_dir / uid
        if p.exists():
            return str(p)
        if self.ext_guess:
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                p2 = self.img_dir / (uid + ext)
                if p2.exists():
                    return str(p2)
        # fallback: try glob
        candidates = list(self.img_dir.glob(uid + '*'))
        if candidates:
            return str(candidates[0])
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])
        path = self._resolve_path(uid)
        if path is None:
            raise FileNotFoundError(f'image for uid {uid} not found in {self.img_dir}')
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label']) if 'label' in row.index else -1
        patient = row.get('patient_id', -1)
        return uid, img, label, patient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', default='.', help='project root')
    parser.add_argument('--img-dir', default='data/cbis_ddsm_demo', help='relative to project-dir or absolute')
    parser.add_argument('--uid-csv', default='data/uid_label_map.csv', help='csv with columns uid,label,patient_id')
    parser.add_argument('--out-dir', default='features', help='where to save features (.npz)')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--model-name', default='resnet18')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    PROJECT_DIR = Path(args.project_dir)
    IMG_DIR = args.img_dir
    UID_CSV = PROJECT_DIR / args.uid_csv if not Path(args.uid_csv).is_absolute() else Path(args.uid_csv)
    IMG_DIR = PROJECT_DIR / IMG_DIR if not Path(IMG_DIR).is_absolute() else Path(IMG_DIR)
    OUT_DIR = PROJECT_DIR / args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(UID_CSV)
    # minimal checks
    if 'uid' not in df.columns:
        print('ERROR: uid column missing in CSV', file=sys.stderr)
        sys.exit(1)
    if 'label' not in df.columns:
        print('WARNING: label column missing — labels will be -1')

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    ds = ImageDataset(df, IMG_DIR, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # device and threading
    device = torch.device(args.device)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    # create backbone that outputs features
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    uids = []
    labels = []
    patients = []
    feats_list = []

    with torch.no_grad():
        for batch in loader:
            batch_uids, imgs, batch_labels, batch_patients = batch
            imgs = imgs.to(device)
            feat = model.forward_features(imgs) if hasattr(model, 'forward_features') else model(imgs)
            # flatten per-sample
            feat = feat.detach().cpu().numpy()
            if feat.ndim > 2:
                feat = feat.reshape(feat.shape[0], -1)
            feats_list.append(feat)
            uids.extend([str(x) for x in batch_uids])
            labels.extend([int(x) for x in batch_labels])
            patients.extend([int(x) if x is not None and x!=-1 else -1 for x in batch_patients])

    feats = np.vstack(feats_list)
    out_path = OUT_DIR / 'features.npz'
    np.savez_compressed(out_path, uids=np.array(uids), features=feats, labels=np.array(labels), patients=np.array(patients))
    print('Saved features:', out_path)


if __name__ == '__main__':
    main()
