from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd
import logging

def generate_drift_report(X_train, X_test, y_train=None, y_test=None, output_path='drift_report.html'):
    """Gera relatório de deriva de dados e alvo usando Evidently."""
    logging.info('Gerando relatório de deriva de dados com Evidently...')
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_df, current_data=test_df)
    report.save_html(output_path)
    logging.info(f'Relatório de deriva salvo em: {output_path}')
    if y_train is not None and y_test is not None:
        target_report = Report(metrics=[TargetDriftPreset()])
        target_report.run(reference_data=pd.DataFrame(y_train), current_data=pd.DataFrame(y_test))
        target_report.save_html('target_' + output_path)
        logging.info(f'Relatório de deriva do alvo salvo em: target_{output_path}') 