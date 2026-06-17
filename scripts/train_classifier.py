#!/usr/bin/env python3
"""
Train a classifier (LightGBM or RandomForest) on precomputed features.
Usage:
  python scripts/train_classifier.py --project-dir . --features-dir features --model out/model.pkl
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', default='.', help='project root')
    parser.add_argument('--features-dir', default='features', help='where features.npz is located')
    parser.add_argument('--out-dir', default='models', help='where to save model and reports')
    parser.add_argument('--clf', default='rf', choices=['rf'], help='classifier')
    parser.add_argument('--n-estimators', type=int, default=200)
    args = parser.parse_args()

    PROJ = Path(args.project_dir)
    FEATURES = PROJ / args.features_dir / 'features.npz'
    OUT = PROJ / args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(FEATURES, allow_pickle=True)
    uids = data['uids']
    X = data['features']
    y = data['labels']
    patients = data['patients']

    # basic split via GroupKFold when possible; fallback to simple cross-val when groups not provided
    clf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=42)
    unique_groups = np.unique(patients)
    if len(unique_groups) > 1:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=min(5, len(unique_groups)))
        scores = []
        for train_idx, test_idx in gkf.split(X, y, groups=patients):
            clf.fit(X[train_idx], y[train_idx])
            scores.append(clf.score(X[test_idx], y[test_idx]))
        print('GroupKFold scores:', scores)
        print('Mean acc:', float(np.mean(scores)))
    else:
        # fallback cross-val
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X, y, cv=5)
        print('Cross-val scores:', scores)
        print('Mean acc:', float(np.mean(scores)))

    # train final on all
    clf.fit(X, y)
    joblib.dump(clf, OUT / 'classifier_rf.joblib')
    print('Saved model to', OUT / 'classifier_rf.joblib')

    preds = clf.predict(X)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)
    import json
    with open(OUT / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.savefig(OUT / 'confusion_matrix.png')
    print('Saved reports in', OUT)


if __name__ == '__main__':
    main()
