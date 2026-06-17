import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

DATA_DIR = Path('data/cbis_ddsm')
JPEG_DIR = DATA_DIR / 'jpeg'
CSV_DIR = DATA_DIR / 'csv'
OUT_DIR = Path('data/cbis_ddsm_images_v2')
REPORTS = Path('reports/image_demo_v2')

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# read all case description CSVs (mass and calc, train and test)
case_files = list(CSV_DIR.glob('*case_description*.csv'))
print('case files:', case_files)

cases = []
for f in case_files:
    try:
        df = pd.read_csv(f)
        df['source_file'] = f.name
        cases.append(df)
    except Exception as e:
        print('skip csv', f, e)

if not cases:
    raise SystemExit('No case CSVs found')

cases_df = pd.concat(cases, ignore_index=True, sort=False)
# Relevant columns: patient_id, pathology (contains MALIGNANT/BENIGN), image file path
print('cases rows:', len(cases_df))

# build mapping from SeriesInstanceUID (the UID present as a path component in image file path) to label
def extract_uids(image_path_str):
    # image_path_str like: Mass-Training_P_00001_LEFT_CC/1.3.6.1.../1.3.6.1.../000000.dcm
    parts = str(image_path_str).split('/')
    # find all parts that look like UIDs (start with '1.3.6')
    uids = [p for p in parts if p.startswith('1.3.6')]
    return uids

uid_label = {}
for _, row in cases_df.iterrows():
    path = row.get('image file path') or row.get('image file path ')
    pathology = str(row.get('pathology','')).strip().upper()
    label = None
    if 'MALIGNANT' in pathology:
        label = 'malignant'
    elif 'BENIGN' in pathology:
        label = 'benign'
    else:
        # fallback to assessment/abnormality
        assess = str(row.get('assessment','')).upper()
        if 'MALIGNANT' in assess:
            label = 'malignant'
        elif 'BENIGN' in assess:
            label = 'benign'
    if path and label:
        uids = extract_uids(path)
        for u in uids:
            uid_label[u] = label

print('Mapped UIDs with labels:', len(uid_label))

# scan JPEG_DIR subfolders and map each image file to label by parent UID
records = []
for series_dir in JPEG_DIR.iterdir():
    if not series_dir.is_dir():
        continue
    uid = series_dir.name
    label = uid_label.get(uid, 'unknown')
    for img in series_dir.iterdir():
        if img.is_file():
            records.append({'path': str(img), 'uid': uid, 'label': label})

print('Found images entries:', len(records))
import pandas as pd
df = pd.DataFrame(records)
# keep only labeled for stratified split
labeled = df[df.label.isin(['benign','malignant'])]
print('Labeled images:', len(labeled), 'Unknown images:', len(df)-len(labeled))

# stratified split by label at image level (could also be by patient)
train, test = train_test_split(labeled, test_size=0.2, stratify=labeled['label'], random_state=42)
train, val = train_test_split(train, test_size=0.125, stratify=train['label'], random_state=42)

splits = {'train': train, 'val': val, 'test': test}
# create symlinks into OUT_DIR
for split, sdf in splits.items():
    for _, row in sdf.iterrows():
        src = Path(row['path'])
        lbl = row['label']
        dst_dir = OUT_DIR / split / lbl
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            try:
                # create relative symlink
                os.symlink(os.path.relpath(src, dst_dir), dst)
            except Exception as e:
                # fallback to copy if symlink fails
                try:
                    shutil.copy2(src, dst)
                except Exception as e2:
                    print('failed to link or copy', src, e, e2)

# save metadata
df.to_csv(OUT_DIR / 'metadata_all.csv', index=False)
for split, sdf in splits.items():
    sdf.to_csv(OUT_DIR / f'metadata_{split}.csv', index=False)

print('Dataset prepared at', OUT_DIR)

# Demo training: sample up to 500 images balanced
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

max_samples = 500
classes = ['benign','malignant']
per_class = max_samples // len(classes)
sampled_paths = []
for c in classes:
    cdf = labeled[labeled.label==c]
    sampled = cdf.sample(n=min(per_class, len(cdf)), random_state=42)['path'].tolist()
    sampled_paths += [(p,c) for p in sampled]

X=[]
y=[]
size=(64,64)
for p,c in sampled_paths:
    try:
        im=Image.open(p).convert('L').resize(size)
        arr=np.array(im).reshape(-1)/255.0
        X.append(arr)
        y.append(1 if c=='malignant' else 0)
    except Exception as e:
        print('skip',p,e)

if len(X)>=10:
    X=np.array(X)
    y=np.array(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    report=classification_report(y_test,y_pred,output_dict=True)
    cm=confusion_matrix(y_test,y_pred)
    pd.DataFrame(report).to_json(REPORTS / 'image_demo_classification_report.json', orient='columns')
    pd.DataFrame(cm).to_csv(REPORTS / 'image_demo_confusion_matrix.csv', index=False)
    try:
        import joblib
        joblib.dump(clf, REPORTS / 'image_demo_rf.joblib')
    except Exception:
        pass
    print('Demo training finished. samples:', len(X))
else:
    print('Not enough samples for demo:', len(X))

print('Done')
