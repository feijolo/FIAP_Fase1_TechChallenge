import os
import shutil
from pathlib import Path
import random
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = Path('data/cbis_ddsm')
OUT_ROOT = Path('data/cbis_ddsm_images')
REPORTS = Path('reports/image_demo')

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def find_images(root):
    imgs = []
    for p in root.rglob('*'):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


def infer_label(path: Path):
    # Basic heuristics: look for 'malignant' or 'benign' in parent dirs or filename
    s = str(path).lower()
    if 'malignant' in s or 'malign' in s or 'cancer' in s:
        return 'malignant'
    if 'benign' in s or 'benignity' in s:
        return 'benign'
    # fallback: try parent dir name
    parent = path.parent.name.lower()
    if 'malignant' in parent or 'malign' in parent:
        return 'malignant'
    if 'benign' in parent:
        return 'benign'
    return 'unknown'


def prepare_structure(imgs, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        for label in ['benign','malignant','unknown']:
            (out_root / split / label).mkdir(parents=True, exist_ok=True)


def make_sample_and_train(df, out_dir: Path, max_samples=500, image_size=(64,64)):
    # take up to max_samples balanced
    df_known = df[df.label.isin(['benign','malignant'])]
    if df_known.empty:
        print('No labeled images found for demo training.')
        return None

    # sample equally
    classes = df_known['label'].unique().tolist()
    per_class = max_samples // len(classes)
    sampled = []
    for c in classes:
        cdf = df_known[df_known.label == c]
        sampled += cdf.sample(n=min(per_class, len(cdf)), random_state=42)['path'].tolist()

    # load images, resize, flatten
    X = []
    y = []
    for p in sampled:
        try:
            img = Image.open(p).convert('L').resize(image_size)
            arr = np.array(img).reshape(-1) / 255.0
            X.append(arr)
            label = 'malignant' if 'mal' in str(p).lower() else 'benign'
            y.append(1 if label == 'malignant' else 0)
        except Exception as e:
            print('skip', p, e)

    X = np.array(X)
    y = np.array(y)

    # quick classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    if len(X) < 10:
        print('Too few samples for demo training:', len(X))
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).to_json(REPORTS / 'image_demo_classification_report.json', orient='columns')
    pd.DataFrame(cm).to_csv(REPORTS / 'image_demo_confusion_matrix.csv', index=False)

    # save model
    try:
        import joblib
        joblib.dump(clf, REPORTS / 'image_demo_rf.joblib')
    except Exception as e:
        print('Could not save model:', e)

    print('Demo training complete. Samples used:', len(X))
    return {'n_samples': len(X), 'report': report}


def main():
    print('Scanning for images in', DATA_ROOT)
    imgs = find_images(DATA_ROOT)
    print('Found', len(imgs), 'image files (extensions:', IMG_EXTS, ')')

    # build dataframe with inferred labels
    records = []
    for p in imgs:
        lbl = infer_label(p)
        records.append({'path': str(p), 'label': lbl})
    df = pd.DataFrame(records)

    # if labels unknown, try to read CSVs
    if (DATA_ROOT / 'metadata.csv').exists():
        print('Found metadata.csv, attempting to merge labels')
        md = pd.read_csv(DATA_ROOT / 'metadata.csv')
        # naive merge on filename
        md['fname'] = md['file_name'].astype(str).apply(lambda x: Path(x).name)
        df['fname'] = df['path'].apply(lambda x: Path(x).name)
        merged = df.merge(md[['fname','label']], on='fname', how='left')
        df['label'] = merged['label'].fillna(df['label'])
        df.drop(columns=['fname'], inplace=True)

    # Prepare output structure
    prepare_structure(imgs, OUT_ROOT)

    # Simple split: stratify by label if possible
    known = df[df.label.isin(['benign','malignant'])]
    unknown = df[~df.label.isin(['benign','malignant'])]

    if not known.empty:
        train, test = train_test_split(known, test_size=0.2, stratify=known['label'], random_state=42)
        train, val = train_test_split(train, test_size=0.125, stratify=train['label'], random_state=42)  # 0.125 of train -> ~0.1 overall
    else:
        train, val, test = train_test_split(df, test_size=0.2, random_state=42), pd.DataFrame(), pd.DataFrame()

    def copy_rows(rows, split):
        for _, r in rows.iterrows():
            src = Path(r['path'])
            lbl = r['label'] if r['label'] in ['benign','malignant'] else 'unknown'
            dst = OUT_ROOT / split / lbl / src.name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print('copy failed', src, e)

    copy_rows(train, 'train')
    copy_rows(val, 'val')
    copy_rows(test, 'test')

    # save metadata CSV
    df.to_csv(OUT_ROOT / 'metadata_all.csv', index=False)
    train.to_csv(OUT_ROOT / 'metadata_train.csv', index=False)
    val.to_csv(OUT_ROOT / 'metadata_val.csv', index=False)
    test.to_csv(OUT_ROOT / 'metadata_test.csv', index=False)

    # Run demo training on subset
    res = make_sample_and_train(df, OUT_ROOT, max_samples=500, image_size=(64,64))
    if res:
        print('Demo training finished, report in', REPORTS)
    else:
        print('Demo training could not run or produced no results.')


if __name__ == '__main__':
    main()
