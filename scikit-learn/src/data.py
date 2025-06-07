from sklearn.datasets import load_iris, load_wine, load_digits, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
import pandas as pd

def load_dataset(name: str = 'iris', test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, str]:
    """Carrega e divide o dataset especificado ('iris', 'wine', 'digits', 'text')."""
    if name == 'iris':
        ds = load_iris()
        X, y = ds.data, ds.target
        target_names = ds.target_names
        data_type = 'tabular'
    elif name == 'wine':
        ds = load_wine()
        X, y = ds.data, ds.target
        target_names = ds.target_names
        data_type = 'tabular'
    elif name == 'digits':
        ds = load_digits()
        X, y = ds.data, ds.target
        target_names = ds.target_names
        data_type = 'image'
    elif name == 'text':
        ds = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.sport.baseball'], remove=('headers', 'footers', 'quotes'))
        X, y = ds.data, ds.target
        target_names = ds.target_names
        data_type = 'text'
    else:
        raise ValueError(f"Dataset não suportado: {name}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, target_names, data_type 