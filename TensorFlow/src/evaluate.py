from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Any

def evaluate_model(model: keras.Model, x_test, y_test) -> float:
    """Avalia o modelo e retorna a acurácia no teste."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_acc

def plot_history(history: Any):
    """Plota as curvas de acurácia de treino e validação."""
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.title('Acurácia durante o treinamento')
    plt.show() 