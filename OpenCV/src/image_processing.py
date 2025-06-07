import cv2
import numpy as np
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def read_image(path: str) -> Optional[np.ndarray]:
    """Lê uma imagem do disco."""
    image = cv2.imread(path)
    if image is None:
        logging.error(f'Não foi possível ler a imagem: {path}')
        return None
    logging.info(f'Imagem lida: {path} (shape={image.shape})')
    return image

def save_image(path: str, image: np.ndarray) -> bool:
    """Salva uma imagem no disco."""
    try:
        cv2.imwrite(path, image)
        logging.info(f'Imagem salva: {path}')
        return True
    except Exception as e:
        logging.error(f'Erro ao salvar imagem: {e}')
        return False

def detect_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Detecta bordas usando o algoritmo de Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    logging.info(f'Bordas detectadas (Canny: low={low}, high={high})')
    return edges

def show_images(images: dict):
    """Exibe múltiplas imagens em janelas separadas."""
    for title, img in images.items():
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 