import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, DeepLift
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def explain_image_prediction(model: torch.nn.Module, input_image: torch.Tensor, target_class: int, method: str = 'IntegratedGradients', device: torch.device = 'cpu'):
    """Gera explicações para a predição de uma imagem usando Captum (IntegratedGradients ou DeepLift)."""
    model.eval()
    model.zero_grad()

    # Certificar-se de que a imagem de entrada tem o formato correto (C, H, W)
    # Captum espera (N, C, H, W) para a maioria dos modelos de visão
    # Nosso modelo MNIST espera (N, 28*28) após o flatten, mas as atribuições são para a entrada original (N, C, H, W)
    # Então, precisamos de um forward_func que lide com o flatten internamente ou adapte a entrada.
    
    # Para simplicidade, vamos usar o IntegratedGradients diretamente no input_image formatado
    # e a função forward da rede. A visualização assumirá um formato de imagem.
    
    # Adaptar o modelo para a camada Flatten
    # Captum precisa de um `forward_func` que retorna os logits/scores da classe alvo
    def custom_forward(input_flat):
        # Redimensiona o input_flat para o formato original da imagem para visualização
        # Isso é uma simplificação. Em um cenário real, o input para explainers
        # seria o tensor da imagem original (N, C, H, W), e a função forward do modelo
        # lidaria com o flatten internamente para a inferência.
        return model(input_flat.view(-1, 28*28))

    if method == 'IntegratedGradients':
        explainer = IntegratedGradients(custom_forward)
    elif method == 'DeepLift':
        explainer = DeepLift(custom_forward)
    else:
        logging.error(f"Método de explicação inválido: {method}")
        return

    # A imagem de entrada para o explainer deve ter o mesmo formato do modelo original (antes do flatten)
    # Vamos remodelar para (1, 1, 28, 28) se for o caso de MNIST (C, H, W)
    # Se o input_image já for 28x28 e sem canal, adicione um canal e um batch
    if input_image.dim() == 2: # Ex: (28, 28)
        input_image_for_explainer = input_image.unsqueeze(0).unsqueeze(0) # Adiciona canal e batch: (1, 1, 28, 28)
    elif input_image.dim() == 3 and input_image.shape[0] == 1: # Ex: (1, 28, 28)
        input_image_for_explainer = input_image.unsqueeze(0) # Adiciona batch: (1, 1, 28, 28)
    else:
        input_image_for_explainer = input_image # Assume que já está no formato (N, C, H, W)

    input_image_for_explainer = input_image_for_explainer.to(device)

    attributions = explainer.attribute(input_image_for_explainer, target=target_class)

    # Visualização
    # Captum viz.visualize_image_attr espera (H, W, C) para imagem RGB ou (H, W) para escala de cinza.
    # Nossos dados MNIST são (1, 28, 28) após o unsqueeze, então precisamos remover o batch e o canal.
    original_image_np = input_image_for_explainer.squeeze().cpu().numpy()
    attributions_np = attributions.squeeze().cpu().numpy()
    
    logging.info(f'Atribuições {method} geradas. Visualizando...')
    _ = viz.visualize_image_attr(attributions_np,
                                original_image_np,
                                method="blended_heat_map",
                                cmap="RdGn",
                                show_colorbar=True,
                                title=f"Atribuições {method} para a classe {target_class}")
    plt.show() 