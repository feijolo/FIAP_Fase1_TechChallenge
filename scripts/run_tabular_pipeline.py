import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import shap
from src.data_utils import load_breast_cancer_from_sklearn


def save_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def generate_correlation_analysis(df, target, reports_dir):
    """Generate correlation heatmap and target correlation bar chart."""
    features = df.drop(columns=[target]).columns.tolist()
    corr_matrix = df[features + [target]].corr()

    # Full correlation heatmap
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Matriz de Correlação — Todas as Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'correlation_heatmap.png'))
    plt.close()

    # Correlation with target
    target_corr = corr_matrix[target].drop(target).sort_values()
    plt.figure(figsize=(10, 10))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in target_corr.values]
    plt.barh(target_corr.index, target_corr.values, color=colors)
    plt.xlabel('Correlação com Target (Diagnóstico)')
    plt.title('Correlação de Pearson — Features vs Diagnóstico')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'target_correlation.png'))
    plt.close()

    return corr_matrix


def generate_shap_analysis(model, X_data, feature_names, model_name, reports_dir):
    """Generate SHAP summary and bar plots for model interpretability."""
    print(f'  Generating SHAP analysis for {model_name}...')

    if model_name == 'rf':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_data)

    shap_values = explainer.shap_values(X_data)

    # For binary classification with TreeExplainer, take class 1 (benign)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # SHAP summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_data, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title(f'SHAP Summary — {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f'shap_summary_{model_name}.png'),
                bbox_inches='tight')
    plt.close()

    # SHAP bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_data, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=15)
    plt.title(f'SHAP Feature Importance — {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f'shap_importance_{model_name}.png'),
                bbox_inches='tight')
    plt.close()

    return shap_vals


def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = load_breast_cancer_from_sklearn()
    target = 'target'
    feature_names = df.drop(columns=[target]).columns.tolist()
    X = df.drop(columns=[target])
    y = df[target]

    # --- Correlation Analysis ---
    print('Generating correlation analysis...')
    corr_matrix = generate_correlation_analysis(df, target, 'reports')

    # --- Preprocessing: StandardScaler ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.joblib')

    # --- Train / Validation / Test Split (60/20/20) ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    print(f'Split sizes — Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}')

    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # --- Cross-Validation on trainval set ---
    print('\nCross-validation (5-fold stratified on train+val):')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X_trainval, y_trainval, cv=cv,
            scoring=['accuracy', 'recall', 'f1', 'roc_auc'],
            return_train_score=False
        )
        cv_results[name] = {
            metric: {
                'mean': float(np.mean(scores[f'test_{metric}'])),
                'std': float(np.std(scores[f'test_{metric}'])),
                'folds': [float(v) for v in scores[f'test_{metric}']]
            }
            for metric in ['accuracy', 'recall', 'f1', 'roc_auc']
        }
        print(f'  {name}: accuracy={cv_results[name]["accuracy"]["mean"]:.4f} '
              f'(±{cv_results[name]["accuracy"]["std"]:.4f}), '
              f'recall={cv_results[name]["recall"]["mean"]:.4f}, '
              f'f1={cv_results[name]["f1"]["mean"]:.4f}, '
              f'roc_auc={cv_results[name]["roc_auc"]["mean"]:.4f}')

    # --- Train final models and evaluate on test set ---
    summary = {'cross_validation': cv_results}
    print('\nTraining final models and evaluating on test set:')

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Validation set evaluation
        y_val_pred = model.predict(X_val)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)

        # Test set evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True,
                                       target_names=['maligno', 'benigno'])
        cm = confusion_matrix(y_test, y_pred)

        print(f'  {name} — Test accuracy: {report["accuracy"]:.4f}, '
              f'Recall(maligno): {report["maligno"]["recall"]:.4f}, '
              f'F1(maligno): {report["maligno"]["f1-score"]:.4f}')

        summary[name] = {
            'classification_report': report,
            'validation_report': val_report,
            'confusion_matrix': cm.tolist()
        }

        # Save model
        joblib.dump(model, f'models/{name}.joblib')

        # Confusion matrix
        save_confusion_matrix(cm, labels=['maligno', 'benigno'],
                              outpath=f'reports/cm_{name}.png')

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC — {name.upper()}')
        plt.legend(loc='lower right')
        plt.savefig(f'reports/roc_{name}.png')
        plt.close()
        summary[name]['roc_auc'] = float(roc_auc)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision, label=f'{name.upper()}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall — {name.upper()}')
        plt.legend()
        plt.savefig(f'reports/pr_curve_{name}.png')
        plt.close()

        # Feature importance for RF (standard)
        if name == 'rf':
            feat_imp = model.feature_importances_.tolist()
            imp = sorted(zip(feature_names, feat_imp),
                         key=lambda x: x[1], reverse=True)[:10]
            summary[name]['top_features'] = imp
            cols, imps = zip(*imp)
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(imps), y=list(cols))
            plt.title('Top 10 Feature Importances (Random Forest)')
            plt.xlabel('Importância')
            plt.tight_layout()
            plt.savefig('reports/feature_importance_rf.png')
            plt.close()

        # --- SHAP Analysis ---
        generate_shap_analysis(model, X_test, feature_names, name, 'reports')

    # --- Metric Choice Discussion ---
    summary['metric_discussion'] = {
        'primary_metric': 'recall',
        'justification': (
            'Em diagnóstico de câncer de mama, a métrica mais crítica é o RECALL '
            '(sensibilidade), pois falsos negativos — não detectar um tumor maligno — '
            'podem ter consequências graves para a paciente, atrasando o tratamento. '
            'Um falso positivo (classificar benigno como maligno) gera exames adicionais, '
            'mas um falso negativo pode ser fatal. Por isso, priorizamos recall para a '
            'classe maligna, complementado por F1-score para balancear com precisão, '
            'e ROC-AUC para avaliação geral da capacidade discriminativa do modelo.'
        ),
        'secondary_metrics': ['f1-score', 'roc_auc', 'accuracy']
    }

    # --- Critical Discussion ---
    summary['critical_discussion'] = {
        'can_be_used_in_practice': (
            'O modelo apresenta resultados promissores (ROC-AUC > 0.99), porém há '
            'limitações importantes a considerar antes de qualquer uso clínico:'
        ),
        'limitations': [
            'Dataset pequeno (569 amostras) — insuficiente para generalização clínica robusta.',
            'Dados de uma única fonte (UCI/Wisconsin) — pode não representar populações diversas.',
            'Features pré-extraídas (FNA digitalizada) — não cobre todo o fluxo diagnóstico real.',
            'Ausência de validação prospectiva em ambiente clínico.',
            'Possível overfitting dado o tamanho reduzido do dataset.'
        ],
        'recommended_use': (
            'O modelo deve ser utilizado EXCLUSIVAMENTE como ferramenta de SUPORTE à '
            'decisão médica, jamais como diagnóstico definitivo. O médico sempre deve ter '
            'a palavra final no diagnóstico. Sugere-se uso como sistema de triagem para '
            'priorizar casos suspeitos, reduzindo o tempo de espera para análise '
            'especializada, sem substituir o julgamento clínico.'
        ),
        'next_steps': [
            'Validação com datasets maiores e mais diversos.',
            'Validação prospectiva em ambiente hospitalar.',
            'Integração com outros dados clínicos (idade, histórico familiar, etc.).',
            'Avaliação de fairness/equidade entre diferentes grupos demográficos.'
        ]
    }

    # Save summary
    with open('reports/tabular_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save dataset sample
    df.sample(frac=0.1, random_state=42).to_csv('reports/dataset_sample.csv', index=False)

    print('\nPipeline completo. Relatórios salvos em reports/.')

if __name__ == '__main__':
    main()
