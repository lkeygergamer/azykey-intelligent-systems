import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import logging

# Importar módulos do projeto PyTorch
from model import Net
from explain import explain_image_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

st.set_page_config(page_title="PyTorch ML Dashboard", layout="centered")
st.title("🧠 PyTorch ML Dashboard (MNIST/CIFAR10)")

st.markdown("--- Seja bem-vindo ao seu classificador de imagens! ---")

@st.cache_resource # Cache o modelo para não recarregá-lo a cada interação
def load_pytorch_model(model_path: str, device: torch.device, input_channels: int, image_size: int):
    """Carrega um modelo PyTorch treinado."""
    try:
        model = Net(input_channels=input_channels, image_size=image_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Coloca o modelo em modo de avaliação
        logging.info(f"Modelo carregado com sucesso de {model_path}")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        logging.error(f"Erro ao carregar o modelo: {e}")
        return None

# Configurações do Streamlit na sidebar
st.sidebar.header("Configurações do Modelo")
model_file = st.sidebar.file_uploader("Faça upload do seu modelo (.pt)", type=["pt"])

dataset_choice = st.sidebar.selectbox("Escolha o Dataset", ('mnist', 'cifar10'))

input_channels = 1 if dataset_choice == 'mnist' else 3
image_size = 28 if dataset_choice == 'mnist' else 32

device_option = st.sidebar.selectbox("Escolha o dispositivo", ('cpu', 'cuda'))
device = torch.device(device_option if torch.cuda.is_available() else 'cpu')

model = None
if model_file is not None:
    # Salvar o arquivo temporariamente para carregar
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.getbuffer())
    model = load_pytorch_model("temp_model.pt", device, input_channels, image_size)
    if model:
        st.sidebar.success("Modelo carregado com sucesso!")

# Pipeline de pré-processamento de imagem
def get_transforms(dataset_name: str):
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'cifar10':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError("Dataset não suportado.")

m_transforms = get_transforms(dataset_choice)

# Carregar imagem para inferência
st.header("Upload de Imagem para Predição")
uploaded_file = st.file_uploader("Escolha uma imagem (.png, .jpg, etc.)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    # Exibir a imagem carregada
    # Converte para escala de cinza apenas se for MNIST (1 canal)
    if dataset_choice == 'mnist':
        image = Image.open(uploaded_file).convert('L') 
    else:
        image = Image.open(uploaded_file).convert('RGB') # CIFAR10 é RGB (3 canais)

    st.image(image, caption='Imagem Carregada', use_column_width=True)
    st.write("")
    st.write("Realizando Predição...")

    # Pré-processar a imagem
    input_tensor = m_transforms(image).unsqueeze(0).to(device) # Adicionar dimensão de batch

    # Fazer a predição
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_proba = torch.max(probabilities).item()

    st.success(f"Predição: Dígito/Classe **{predicted_class}** com {predicted_proba:.2f}% de confiança")

    # Gerar Explicação com Captum
    st.subheader("Explicabilidade da Predição")
    if st.button("Gerar Explicação (IntegratedGradients)"):
        if predicted_class is not None:
            with st.spinner("Gerando explicações... Isso pode levar um momento."):
                # A função explain_image_prediction já lida com a exibição do plot
                explain_image_prediction(model, input_tensor.squeeze(0), predicted_class, device=device)
                st.success("Explicação gerada! Verifique a janela de plotagem.")
        else:
            st.warning("Por favor, faça uma predição primeiro para gerar a explicação.")

elif uploaded_file is None and model is not None:
    st.info("Faça upload de uma imagem para ver a predição!")
elif model is None:
    st.warning("Por favor, faça upload de um modelo PyTorch (.pt) na sidebar para começar!")

st.sidebar.markdown("--- ")
st.sidebar.info("Desenvolvido com PyTorch e Streamlit. Blocos de construção para AGI!") 