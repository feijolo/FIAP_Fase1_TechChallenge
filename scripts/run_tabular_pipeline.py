import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from src.data_utils import load_breast_cancer_from_sklearn


def save_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = load_breast_cancer_from_sklearn()
    target = 'target'
    X = df.drop(columns=[target])
    y = df[target]

    # simple preprocessing: scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    summary = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        summary[name] = {
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        # save model
        try:
            import joblib
            joblib.dump(model, f'models/{name}.joblib')
        except Exception as e:
            print('joblib save failed:', e)

        # save confusion matrix image
        save_confusion_matrix(cm, labels=['benign','malignant'], outpath=f'reports/cm_{name}.png')

        # ROC curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - {name}')
            plt.legend(loc='lower right')
            plt.savefig(f'reports/roc_{name}.png')
            plt.close()
            summary[name]['roc_auc'] = float(roc_auc)

        # feature importance for RF
        if name == 'rf':
            feat_imp = model.feature_importances_.tolist()
            columns = df.drop(columns=[target]).columns.tolist()
            imp = sorted(zip(columns, feat_imp), key=lambda x: x[1], reverse=True)[:10]
            summary[name]['top_features'] = imp
            # plot
            cols, imps = zip(*imp)
            plt.figure(figsize=(6,4))
            sns.barplot(x=list(imps), y=list(cols))
            plt.title('Top 10 feature importances (RF)')
            plt.tight_layout()
            plt.savefig('reports/feature_importance_rf.png')
            plt.close()

    # save summary
    with open('reports/tabular_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # save a small sample of the dataset for record
    df.sample(frac=0.1, random_state=42).to_csv('reports/dataset_sample.csv', index=False)

    print('Completed tabular pipeline. Reports saved to reports/*.')

if __name__ == '__main__':
    main()
