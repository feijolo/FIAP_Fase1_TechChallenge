# Small helper to download datasets; for Kaggle-based datasets, place kaggle.json in ~/.kaggle/
import os
import subprocess


def download_kaggle(dataset, dest='data'):
    os.makedirs(dest, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', dest, '--unzip']
    subprocess.check_call(cmd)


if __name__ == '__main__':
    # Example: download_kaggle('uciml/breast-cancer-wisconsin-data')
    print('Run as script or import functions.download_kaggle')
