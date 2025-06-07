from tensorflow import keras
from tensorflow.keras import layers, Sequential

def build_model(input_shape=(28, 28), num_classes=10) -> keras.Model:
    """Cria e retorna um modelo sequencial Keras para MNIST."""
    model = Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model 