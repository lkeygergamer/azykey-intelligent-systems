import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net, OptimizedNet
from train import train_model
from evaluate import evaluate_model
from data import get_data_loaders
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def objective(trial: optuna.Trial, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, input_channels: int, image_size: int) -> float:
    """Função objetivo para otimização de hiperparâmetros com Optuna."""
    # Sugere hiperparâmetros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    n_hidden1 = trial.suggest_int('n_hidden1', 64, 256)
    n_hidden2 = trial.suggest_int('n_hidden2', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Cria o modelo com os hiperparâmetros sugeridos, passando as dimensões corretas
    model = OptimizedNet(n_hidden1, n_hidden2, dropout_rate, input_channels=input_channels, image_size=image_size).to(device)
    
    # Treina o modelo
    train_model(model, train_loader, device, epochs=trial.suggest_int('epochs', 2, 5), lr=lr, checkpoint_path=None) # Sem checkpoints para otimização rápida

    # Avalia o modelo
    accuracy = evaluate_model(model, test_loader, device)
    return accuracy

def run_automl_optuna(train_loader: DataLoader, test_loader: DataLoader, device: torch.device, n_trials: int = 20, input_channels: int = 1, image_size: int = 28) -> optuna.study.Study:
    """Executa a otimização AutoML com Optuna."""
    logging.info(f'Iniciando AutoML com Optuna para {n_trials} tentativas...')
    study = optuna.create_study(direction='maximize')
    # Passa input_channels e image_size para a função objetivo
    study.optimize(lambda trial: objective(trial, train_loader, test_loader, device, input_channels, image_size), n_trials=n_trials)
    logging.info('AutoML com Optuna finalizado.')
    logging.info(f'Melhores parâmetros: {study.best_trial.params}')
    logging.info(f'Melhor acurácia: {study.best_trial.value:.4f}%')
    return study 