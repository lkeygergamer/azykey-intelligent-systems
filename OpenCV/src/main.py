import argparse
import logging
from image_processing import read_image, save_image, detect_edges, show_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Processamento avançado de imagens com OpenCV')
    parser.add_argument('--input', required=True, help='Caminho da imagem de entrada')
    parser.add_argument('--output', default='output_edges.jpg', help='Caminho para salvar a imagem de bordas')
    parser.add_argument('--low', type=int, default=100, help='Limite inferior para Canny')
    parser.add_argument('--high', type=int, default=200, help='Limite superior para Canny')
    args = parser.parse_args()

    image = read_image(args.input)
    if image is None:
        return
    edges = detect_edges(image, args.low, args.high)
    save_image(args.output, edges)
    show_images({'Original': image, 'Bordas (Canny)': edges})

if __name__ == '__main__':
    main() 