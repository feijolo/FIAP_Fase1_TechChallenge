import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def train_tabular_models(df, target='target'):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'model': model,
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        joblib.dump(model, f'models/{name}.joblib')

    return results


if __name__ == '__main__':
    import os
    import pandas as pd
    from data_utils import load_breast_cancer_from_sklearn

    os.makedirs('models', exist_ok=True)
    df = load_breast_cancer_from_sklearn()
    res = train_tabular_models(df)
    print(res.keys())
