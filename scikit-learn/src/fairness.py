from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def analyze_fairness(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray, protected_attribute: pd.Series, privileged_groups: list, unprivileged_groups: list, label_names=None) -> dict:
    """Analisa a imparcialidade do modelo usando AIF360."""
    logging.info('Iniciando análise de imparcialidade com AIF360...')

    # Prepare os dados para AIF360
    df_test = pd.DataFrame(X_test)
    df_test['target'] = y_test
    df_test[protected_attribute.name] = protected_attribute.values # Adiciona o atributo protegido ao DataFrame

    # Preveja com o pipeline
    y_pred_proba = pipeline.predict_proba(X_test)
    # Assumindo que a classe positiva é a última coluna na saída do predict_proba
    positive_class_idx = 1 # Para classificação binária, ou a classe de interesse
    if y_pred_proba.shape[1] > 2: # Para multiclasse, precisamos de uma estratégia mais robusta
        logging.warning("Detecção de bias AIF360 para multiclasse é complexa. Assumindo classe 1 como positiva.")
        # Uma abordagem real poderia ser one-vs-rest ou analisar cada classe

    # AIF360 BinaryLabelDataset espera rótulos 0 e 1, e atributo protegido 0 e 1
    # Mapear os grupos privilegiados/não privilegiados para valores binários (ex: 1 e 0)
    # Esta parte é genérica e pode precisar de ajuste dependendo dos seus dados reais
    # protected_attribute_map = {group_val: binary_val for binary_val, group_val in enumerate(privileged_groups + unprivileged_groups)}
    # mapped_protected_attribute = protected_attribute.map(protected_attribute_map).fillna(-1).astype(int) # -1 para valores não mapeados

    # Para simplificar, AIF360 tipicamente trabalha com a coluna protegida sendo 0 ou 1
    # E os grupos privilegiados/não privilegiados são baseados nesses valores
    # Aqui, vamos usar a codificação original do atributo protegido para os grupos
    # E assumir que o modelo prevê a classe binária 0 ou 1

    # AIF360 requer que os grupos sejam listas de dicionários
    # Ex: privileged_groups=[{'sex': 1}], unprivileged_groups=[{'sex': 0}]
    # Vamos simplificar aqui e assumir que o 'protected_attribute' é binário 0/1
    # E que privileged_groups é [value_for_privileged], unprivileged_groups é [value_for_unprivileged]

    # Recrie privileged_groups e unprivileged_groups no formato que AIF360 espera
    # Isso é um hack para o exemplo, em produção você passaria o nome da coluna real
    # e os valores reais dos grupos
    protected_attribute_name = protected_attribute.name
    privileged_groups_aif = [{protected_attribute_name: val} for val in privileged_groups]
    unprivileged_groups_aif = [{protected_attribute_name: val} for val in unprivileged_groups]

    # O dataset deve ser construído com os dados para AIF360
    # Recomenda-se que o atributo protegido seja uma coluna no DataFrame
    # E que os rótulos preditos sejam binários 0 ou 1
    
    # Para ClassificationMetric, o dataset AIF360 precisa ter as colunas
    # 'protected_attribute_name', 'target', 'predicted_labels'
    # e 'scores' (para predict_proba)

    # Criando o DataFrame para AIF360
    df_aif = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    df_aif['target'] = y_test
    df_aif['y_pred_proba'] = y_pred_proba[:, positive_class_idx] # Probabilidade da classe positiva
    df_aif['y_pred_labels'] = np.argmax(y_pred_proba, axis=1) # Rótulos preditos
    df_aif[protected_attribute_name] = protected_attribute.values

    # AIF360 BinaryLabelDataset
    # Aqui assumimos que o atributo protegido é binário (0 ou 1) e que os rótulos também são
    # Isso pode exigir pré-processamento adicional para datasets reais
    bld = BinaryLabelDataset(df=df_aif,
                             label_names=['target'],
                             protected_attribute_names=[protected_attribute_name],
                             privileged_classes=[1],
                             unprivileged_classes=[0])

    # Metric para o modelo sem viés (ideal)
    metric_orig_model = ClassificationMetric(bld,
                                             bld,
                                             unprivileged_groups=unprivileged_groups_aif,
                                             privileged_groups=privileged_groups_aif)
    
    # Resultados
    results = {
        "statistical_parity_difference": metric_orig_model.statistical_parity_difference(),
        "disparate_impact": metric_orig_model.disparate_impact(),
        "average_odds_difference": metric_orig_model.average_odds_difference()
    }
    logging.info(f'Resultados da análise de imparcialidade: {results}')
    return results 