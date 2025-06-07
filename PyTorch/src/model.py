import torch.nn as nn
import torch

class Net(nn.Module):
    """Rede neural simples para classificação de dígitos MNIST ou imagens CIFAR10."""
    def __init__(self, input_channels: int = 1, image_size: int = 28):
        super(Net, self).__init__()
        # Calcule o tamanho de entrada flattenado
        flattened_size = input_channels * image_size * image_size

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10) # 10 classes para MNIST e CIFAR10
        self.dropout = nn.Dropout(0.2)
        self.input_channels = input_channels
        self.image_size = image_size

    def forward(self, x):
        # Redimensiona o input para o formato flattenado
        x = x.view(-1, self.input_channels * self.image_size * self.image_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OptimizedNet(nn.Module):
    """Versão otimizada da rede neural, usada pelo Optuna."""
    def __init__(self, n_hidden1, n_hidden2, dropout_rate, input_channels: int = 1, image_size: int = 28):
        super().__init__()
        flattened_size = input_channels * image_size * image_size

        self.fc1 = nn.Linear(flattened_size, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 10)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_channels = input_channels
        self.image_size = image_size

    def forward(self, x):
        x = x.view(-1, self.input_channels * self.image_size * self.image_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x 