import pandas as pd


def load_breast_cancer_from_sklearn():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data.data, data.target.rename('target')], axis=1)
    return df


if __name__ == '__main__':
    df = load_breast_cancer_from_sklearn()
    print(df.shape)
