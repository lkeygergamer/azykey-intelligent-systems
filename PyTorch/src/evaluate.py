import torch
from torch.utils.data import DataLoader
from model import Net

def evaluate_model(model: Net, test_loader: DataLoader, device: torch.device) -> float:
    """Avalia o modelo e retorna a acurácia."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    return acc 