import argparse
import logging
import joblib
import yaml
import pandas as pd # Import pandas for protected_attribute
from data import load_dataset
from model import build_pipeline
from train import cross_validate_pipeline, train_pipeline
from evaluate import evaluate_pipeline, plot_confusion_matrix
from optimize import optimize_hyperparameters
from report import generate_html_report
from automl import run_automl
from explain import explain_with_shap, explain_with_lime
from monitor import generate_drift_report
from nlg import generate_nlg_report
from deploy import deploy_model_to_local_endpoint
from fairness import analyze_fairness # Import the new module
from sklearn.metrics import confusion_matrix
import numpy as np # Adicionado para np.unique em classes para partial_fit

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description='Pipeline completo de ML multimodal com Aprendizado Contínuo (AGI building block)')
    # Modificado para aceitar múltiplos datasets
    parser.add_argument('--datasets', type=str, default=config.get('datasets', 'iris'), help='Lista de datasets a serem utilizados (separados por vírgula: iris,wine,digits,text)')
    parser.add_argument('--cv', type=int, default=config.get('cv', 5), help='Número de folds para validação cruzada')
    parser.add_argument('--model-path', type=str, default=config.get('model_path', 'pipeline.joblib'), help='Caminho para salvar o pipeline treinado')
    parser.add_argument('--plot-cm', action='store_true' if config.get('plot_cm', False) else 'store_false', help='Exibir matriz de confusão após avaliação')
    parser.add_argument('--test-size', type=float, default=config.get('test_size', 0.2), help='Proporção do conjunto de teste')
    parser.add_argument('--random-state', type=int, default=config.get('random_state', 42), help='Seed para divisão dos dados')
    parser.add_argument('--optimize', action='store_true', help='Executar otimização de hiperparâmetros (GridSearchCV)')
    parser.add_argument('--report', action='store_true', help='Gerar relatório HTML automático')
    parser.add_argument('--automl', action='store_true', help='Executar AutoML com TPOT')
    parser.add_argument('--explain', action='store_true', help='Gerar explicações SHAP e LIME para o modelo')
    parser.add_argument('--monitor', action='store_true', help='Gerar relatório de deriva de dados/modelo (Evidently)')
    parser.add_argument('--nlg', action='store_true', help='Gerar relatório em linguagem natural (OpenAI)')
    parser.add_argument('--deploy', action='store_true', help='Simular deploy do modelo treinado')
    parser.add_argument('--online-learning', action='store_true', help='Habilitar aprendizado contínuo (online learning)')
    parser.add_argument('--fairness', action='store_true', help='Executar análise de imparcialidade (AIF360)')
    parser.add_argument('--augment-text', action='store_true', help='Aumentar dados de texto com LLM (requer --dataset text)')
    args = parser.parse_args()

    # Processar múltiplos datasets para aprendizado contínuo
    dataset_names = [d.strip() for d in args.datasets.split(',')]
    all_data = [] # Para armazenar dados de todos os datasets para avaliação contínua
    
    pipeline = None # Inicializa o pipeline fora do loop

    # --- Início: Coletar todas as classes possíveis para Aprendizado Contínuo ---
    all_possible_classes = []
    if args.online_learning:
        logging.info("Coletando todas as classes possíveis para aprendizado contínuo...")
        for dataset_name in dataset_names:
            _, _, y_train_temp, _, _, _ = load_dataset(name=dataset_name, test_size=0, random_state=args.random_state) # Carrega só o y_train
            all_possible_classes.extend(np.unique(y_train_temp).tolist())
        all_possible_classes = np.array(sorted(list(set(all_possible_classes))))
        logging.info(f"Todas as classes identificadas para aprendizado contínuo: {all_possible_classes}")
    # --- Fim: Coletar todas as classes possíveis para Aprendizado Contínuo ---

    for i, dataset_name in enumerate(dataset_names):
        logging.info(f"\n{'='*50}\nProcessando dataset: {dataset_name} (Passo {i+1}/{len(dataset_names)})\n{'='*50}")
        X_train, X_test, y_train, y_test, target_names, data_type = load_dataset(
            name=dataset_name, test_size=args.test_size, random_state=args.random_state)
        
        # Para aprendizado contínuo, precisamos manter o histórico de dados
        all_data.append({
            'X_test': X_test,
            'y_test': y_test,
            'target_names': target_names,
            'data_type': data_type,
            'dataset_name': dataset_name
        })

        # Aumento de dados de texto com LLM
        if args.augment_text and data_type == 'text':
            from llm_utils import generate_synthetic_text_data
            logging.info("Aplicando aumento de dados de texto com LLM...")
            synthetic_texts, synthetic_labels = generate_synthetic_text_data(X_train.tolist(), y_train.tolist())
            X_train = np.array(list(X_train) + synthetic_texts)
            y_train = np.array(list(y_train) + synthetic_labels)
            logging.info(f"Novos dados de treino após aumento: {len(X_train)} amostras.")

        if args.automl:
            score, export_path, model_path = run_automl(X_train, y_train, X_test, y_test, cv=args.cv)
            logging.info(f'AutoML finalizado para {dataset_name}. Acurácia: {score:.4f}. Pipeline salvo em {model_path}. Código em {export_path}.')
            # Se usarmos AutoML, o pipeline é encontrado e treinado aqui, então não treinamos novamente abaixo
            # Precisamos carregar o pipeline treinado pelo TPOT para as próximas etapas
            pipeline = joblib.load(model_path) 
            logging.info(f'Pipeline carregado do AutoML para {dataset_name}.')
            
        else:
            # Apenas constrói o pipeline na primeira iteração ou se não otimizar
            if pipeline is None or not args.online_learning: # Reconstruir se não for online learning ou primeira vez
                pipeline = build_pipeline(n_estimators=config.get('n_estimators', 100), data_type=data_type, online_mode=args.online_learning)

            if not args.online_learning: # Validação cruzada apenas para offline learning
                acc = cross_validate_pipeline(pipeline, X_train, y_train, cv=args.cv)
                logging.info(f'Acurácia média (validação cruzada) para {dataset_name}: {acc:.2f}')
            
            # Treina o pipeline (fit completo ou partial_fit)
            pipeline = train_pipeline(pipeline, X_train, y_train, online_mode=args.online_learning, all_seen_classes=all_possible_classes)
        
        # Avaliar e logar o desempenho no dataset atual
        report = evaluate_pipeline(pipeline, X_test, y_test, target_names=target_names)
        print(f'Relatório de classificação para {dataset_name} (teste):\n{report}')

        # Avaliar o desempenho em TODOS os datasets anteriores (para Continual Learning)
        if args.online_learning and i > 0: # Apenas a partir do segundo dataset
            logging.info("Avaliação de desempenho em datasets anteriores (esquecimento):")
            for prev_data in all_data[:-1]: # Exclui o dataset atual
                prev_X_test = prev_data['X_test']
                prev_y_test = prev_data['y_test']
                prev_target_names = prev_data['target_names']
                prev_dataset_name = prev_data['dataset_name']
                prev_report = evaluate_pipeline(pipeline, prev_X_test, prev_y_test, target_names=prev_target_names)
                logging.info(f'  Desempenho em {prev_dataset_name}:\n{prev_report}')

        # Geração de artefatos para o dataset atual (se não for AutoML ou no final)
        if not args.automl or i == len(dataset_names) - 1: # Gerar se não for AutoML ou na última iteração do AutoML
            current_model_path = args.model_path.replace('.joblib', f'_{dataset_name}.joblib')
            joblib.dump(pipeline, current_model_path)
            logging.info(f'Pipeline treinado salvo para {dataset_name} em {current_model_path}')

            if args.plot_cm:
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                plot_confusion_matrix(pipeline, X_test, y_test, target_names=target_names)

            if args.report:
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                generate_html_report(report, cm, target_names, output_path=f'report_{dataset_name}.html')

            if args.explain:
                feature_names = None  # Pode ser melhorado para pegar nomes reais
                try:
                    explain_with_shap(pipeline, X_train, X_test, feature_names=feature_names)
                except Exception as e:
                    logging.warning(f'Falha ao gerar explicação SHAP para {dataset_name}: {e}')
                try:
                    explain_with_lime(pipeline, X_train, X_test, feature_names=feature_names, class_names=target_names)
                except Exception as e:
                    logging.warning(f'Falha ao gerar explicação LIME para {dataset_name}: {e}')

            if args.monitor:
                generate_drift_report(X_train, X_test, y_train, y_test, output_path=f'drift_report_{dataset_name}.html')

            if args.nlg:
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                nlg_report = generate_nlg_report(report, cm, target_names)
                if nlg_report:
                    print(f"\n===== Relatório em Linguagem Natural (NLG) para {dataset_name} =====\n")
                    print(nlg_report)
            
            if args.fairness:
                if dataset_name == 'iris': # Apenas para Iris neste exemplo
                    protected_attribute_data = (X_test[:, 0] > X_test[:, 0].mean()).astype(int)
                    protected_attribute = pd.Series(protected_attribute_data, name='simulated_group')
                    privileged_groups = [1] # Ex: valores acima da média
                    unprivileged_groups = [0] # Ex: valores abaixo da média
                    logging.info(f"Executando análise de imparcialidade para {dataset_name} (atributo simulado).")
                    fairness_results = analyze_fairness(pipeline, X_test, y_test, protected_attribute, privileged_groups, unprivileged_groups, label_names=target_names)
                    logging.info(f"Resultados de imparcialidade para {dataset_name}: {fairness_results}")
                else:
                    logging.warning(f"Análise de imparcialidade não configurada para o dataset {dataset_name} neste exemplo. Adapte para outros datasets.")

    # Deploy final (opcional, se houver um pipeline treinado e não for AutoML)
    if args.deploy and pipeline is not None and not args.automl:
        deploy_model_to_local_endpoint(args.model_path)

if __name__ == '__main__':
    main() 