import argparse
import logging
import tensorflow as tf
from data import load_mnist_data
from model import build_model
from train import train_model
from evaluate import evaluate_model, plot_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Treinamento e avaliação de rede neural com TensorFlow/Keras (MNIST)')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--model-path', type=str, default='mnist_model.h5', help='Caminho para salvar o modelo treinado')
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_mnist_data()
    model = build_model()

    history = train_model(model, x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    acc = evaluate_model(model, x_test, y_test)
    logging.info(f'Acurácia no teste: {acc:.2f}')

    model.save(args.model_path)
    logging.info(f'Modelo salvo em {args.model_path}')

    plot_history(history)

if __name__ == '__main__':
    main() 