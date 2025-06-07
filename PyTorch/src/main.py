import argparse
import logging
import torch
import torch.nn as nn
from data import get_data_loaders
from model import Net
from train import train_model
from evaluate import evaluate_model
from optimize import run_automl_optuna
from explain import explain_image_prediction
from monitor import generate_drift_report
from nlg import generate_nlg_report
from deploy import deploy_model_to_local_endpoint
from fairness import analyze_fairness
from llm_utils import augment_data_with_llm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Treinamento e avaliação de rede neural com PyTorch (Multimodal Imagem) com AutoML, Explicabilidade, Monitoramento, NLG, Deploy, Aprendizado Contínuo, Fairness e Aumento de Dados com LLM')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset a ser utilizado (mnist ou cifar10)')
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Taxa de aprendizado')
    parser.add_argument('--checkpoint', type=str, default=None, help='Prefixo para salvar checkpoints do modelo')
    parser.add_argument('--automl', action='store_true', help='Executar AutoML com Optuna para otimização de hiperparâmetros')
    parser.add_argument('--n-trials', type=int, default=20, help='Número de tentativas para o Optuna AutoML')
    parser.add_argument('--explain', action='store_true', help='Gerar explicações para uma predição de imagem usando Captum')
    parser.add_argument('--sample-idx', type=int, default=0, help='Índice da amostra de teste para explicação (se --explain ativado)')
    parser.add_argument('--monitor', action='store_true', help='Gerar relatório de deriva de dados/modelo (Evidently)')
    parser.add_argument('--nlg', action='store_true', help='Gerar relatório em linguagem natural (NLG) do desempenho do modelo')
    parser.add_argument('--deploy', action='store_true', help='Simular o deploy do modelo para um endpoint local')
    parser.add_argument('--online-learning', action='store_true', help='Ativar modo de aprendizado contínuo (online learning)')
    parser.add_argument('--fairness', action='store_true', help='Realizar análise de justiça (fairness) do modelo')
    parser.add_argument('--augment-text', action='store_true', help='Gerar texto sintético para aumento de dados usando LLM')
    args = parser.parse_args()

    # Determinar parâmetros da imagem com base no dataset
    input_channels = 1 if args.dataset == 'mnist' else 3
    image_size = 28 if args.dataset == 'mnist' else 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(dataset_name=args.dataset, batch_size=args.batch_size)
    
    model = None

    if args.automl:
        logging.info("Iniciando modo AutoML (Optuna)...")
        study = run_automl_optuna(train_loader, test_loader, device, n_trials=args.n_trials, 
                                  input_channels=input_channels, image_size=image_size)
        # O melhor modelo é retreinado com os melhores parâmetros encontrados
        best_params = study.best_trial.params
        logging.info(f"Retreinando o modelo com os melhores parâmetros do AutoML: {best_params}")
        
        # Reconstruir o modelo com os melhores parâmetros do Optuna, passando as dimensões corretas
        class BestNet(Net):
            def __init__(self, n_hidden1=best_params.get('n_hidden1', 128), n_hidden2=best_params.get('n_hidden2', 64), 
                         dropout_rate=best_params.get('dropout_rate', 0.2), 
                         input_channels=input_channels, image_size=image_size):
                super().__init__(input_channels, image_size)
                self.fc1 = nn.Linear(input_channels * image_size * image_size, n_hidden1)
                self.fc2 = nn.Linear(n_hidden1, n_hidden2)
                self.fc3 = nn.Linear(n_hidden2, 10)
                self.dropout = nn.Dropout(dropout_rate)

        model = BestNet().to(device) # Usar BestNet com parâmetros otimizados
        train_model(model, train_loader, device, epochs=best_params.get('epochs', 5), lr=best_params['lr'], checkpoint_path=args.checkpoint, online_learning=args.online_learning)

    else:
        model = Net(input_channels=input_channels, image_size=image_size).to(device) # Passar dimensoes para Net
        train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr, checkpoint_path=args.checkpoint, online_learning=args.online_learning)

    acc = evaluate_model(model, test_loader, device)
    logging.info(f'Acurácia final no teste: {acc:.2f}%')

    # Salvar modelo final
    model_filename = f'{args.dataset}_model_final.pt'
    torch.save(model.state_dict(), model_filename)
    logging.info(f'Modelo final salvo em {model_filename}')

    if args.explain:
        logging.info(f"Gerando explicações para a amostra {args.sample_idx}...")
        # Pegar uma amostra do dataset de teste para explicar
        # É importante que o modelo esteja no modo de avaliação (eval) para explicar
        model.eval()
        
        # Obter uma amostra de teste e seu rótulo real
        sample_data, sample_target = test_loader.dataset[args.sample_idx]
        sample_data = sample_data.to(device) # Mover para o dispositivo

        # Prever a classe da amostra
        with torch.no_grad():
            output = model(sample_data.unsqueeze(0)) # Adicionar dimensão de batch
            predicted_class = torch.argmax(output).item()

        logging.info(f"Amostra {args.sample_idx}: Classe Real = {sample_target}, Classe Predita = {predicted_class}")
        explain_image_prediction(model, sample_data, predicted_class, device=device)

    if args.monitor:
        logging.info("Gerando relatórios de deriva de dados e alvo...")
        generate_drift_report(train_loader, test_loader)

    if args.nlg:
        logging.info("Gerando relatório em linguagem natural...")
        nlg_report = generate_nlg_report(acc, args.dataset, "rede neural de imagem")
        print("\n--- Relatório NLG ---")
        print(nlg_report)
        print("---------------------")

    if args.deploy:
        logging.info("Simulando deploy do modelo...")
        deploy_model_to_local_endpoint(model_filename, f'{args.dataset}_image_classifier')

    if args.fairness:
        logging.info("Realizando análise de justiça (fairness)...")
        analyze_fairness(model, test_loader, device, args.dataset)

    if args.augment_text:
        logging.info("Gerando texto sintético para aumento de dados...")
        augment_data_with_llm(args.dataset, num_samples=2) # Gerar 2 exemplos de texto para cada classe

if __name__ == '__main__':
    main() 