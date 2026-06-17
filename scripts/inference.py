#!/usr/bin/env python3
"""
Run inference using saved classifier on features.npz
Usage:
  python scripts/inference.py --project-dir . --features-dir features --model models/classifier_rf.joblib --out reports/predictions.csv
"""
import argparse
from pathlib import Path
import numpy as np
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', default='.')
    parser.add_argument('--features-dir', default='features')
    parser.add_argument('--model', default='models/classifier_rf.joblib')
    parser.add_argument('--out', default='reports/predictions.csv')
    args = parser.parse_args()

    PROJ = Path(args.project_dir)
    data = np.load(PROJ / args.features_dir / 'features.npz', allow_pickle=True)
    uids = data['uids']
    X = data['features']
    y = data['labels']

    clf = joblib.load(PROJ / args.model)
    preds = clf.predict(X)
    probs = clf.predict_proba(X) if hasattr(clf, 'predict_proba') else None

    df = pd.DataFrame({'uid': uids, 'label': y, 'pred': preds})
    if probs is not None:
        # for binary, take prob of class 1
        if probs.shape[1] == 2:
            df['prob_pos'] = probs[:,1]
        else:
            for i in range(probs.shape[1]):
                df[f'prob_{i}'] = probs[:,i]
    df.to_csv(PROJ / args.out, index=False)
    print('Saved predictions to', PROJ / args.out)


if __name__ == '__main__':
    main()
