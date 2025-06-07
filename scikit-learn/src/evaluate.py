from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_pipeline(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray, target_names=None) -> str:
    """Avalia o pipeline e retorna o relatório de classificação."""
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names)
    return report

def plot_confusion_matrix(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray, target_names=None, figsize=(6,5)):
    """Plota a matriz de confusão de forma visual e interativa."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.show() 