from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def generate_drift_report(train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, output_path='drift_report.html'):
    """Gera relatório de deriva de dados e alvo para dados de imagem usando Evidently."""
    logging.info('Gerando relatório de deriva de dados com Evidently para PyTorch...')

    # Coletar dados de referência (treino) e dados atuais (teste)
    # Converter tensores PyTorch para numpy e depois para pandas DataFrame
    # Flatten imagens para formato tabular para Evidently
    
    ref_data_features = []
    ref_data_targets = []
    for data, target in train_loader:
        ref_data_features.append(data.view(data.size(0), -1).cpu().numpy())
        ref_data_targets.append(target.cpu().numpy())
    ref_df = pd.DataFrame(np.vstack(ref_data_features))
    ref_df['target'] = np.hstack(ref_data_targets)

    current_data_features = []
    current_data_targets = []
    for data, target in test_loader:
        current_data_features.append(data.view(data.size(0), -1).cpu().numpy())
        current_data_targets.append(target.cpu().numpy())
    current_df = pd.DataFrame(np.vstack(current_data_features))
    current_df['target'] = np.hstack(current_data_targets)

    # Renomear colunas para evitar conflitos (Evidently pode reclamar de nomes numéricos)
    ref_df.columns = [f'feature_{i}' for i in range(ref_df.shape[1]-1)] + ['target']
    current_df.columns = [f'feature_{i}' for i in range(current_df.shape[1]-1)] + ['target']

    # Relatório de Deriva de Dados
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=ref_df, current_data=current_df)
    data_drift_report.save_html(output_path)
    logging.info(f'Relatório de deriva de dados salvo em: {output_path}')

    # Relatório de Deriva de Alvo (Target Drift)
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=ref_df, current_data=current_df)
    target_drift_report.save_html('target_' + output_path)
    logging.info(f'Relatório de deriva do alvo salvo em: target_{output_path}') 