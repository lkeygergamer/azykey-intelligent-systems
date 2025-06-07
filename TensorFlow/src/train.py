from tensorflow import keras
from typing import Any, Dict

def train_model(model: keras.Model, x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, patience=2) -> Any:
    """Treina o modelo com early stopping e retorna o histórico."""
    early_stop = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop]
    )
    return history 