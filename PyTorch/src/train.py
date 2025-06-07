import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from model import Net

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_model(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int, lr: float, checkpoint_path: str = None, online_learning: bool = False):
    """Treina o modelo com base no loader de treino.
    Suporta treinamento em batch tradicional ou simulação de aprendizado contínuo (online).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if online_learning:
        logging.info("Iniciando treinamento em modo de Aprendizado Contínuo (Online Learning)...")
        # Em aprendizado online, tipicamente fazemos uma única passagem pelos dados,
        # ou um número muito pequeno de épocas, atualizando o modelo a cada batch.
        # Vamos simular uma única passagem por cada dado disponível, como se fosse um stream.
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                logging.info(f'Online Learning: Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}')
        logging.info("Treinamento em modo de Aprendizado Contínuo finalizado.")

    else:
        logging.info(f"Iniciando treinamento por {epochs} épocas...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            logging.info(f'Época {epoch+1}/{epochs}, Perda: {avg_loss:.4f}')

            if checkpoint_path:
                torch.save(model.state_dict(), f'{checkpoint_path}_epoch_{epoch+1}.pt')
                logging.info(f'Checkpoint salvo para época {epoch+1}') 