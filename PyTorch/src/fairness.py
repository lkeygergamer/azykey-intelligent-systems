import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Libraries for fairness analysis (although more suited for tabular data, we'll demonstrate concepts)
# from aif360.datasets import BinaryLabelDataset
# from aif360.metrics import ClassificationMetric
# from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def analyze_fairness(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device, dataset_name: str):
    """Analisa métricas de justiça e viés em um modelo PyTorch.
    
    Para datasets de imagem como MNIST/CIFAR-10 que não têm atributos protegidos explícitos,
    esta função demonstrará a análise de performance por classe como um proxy para grupos,
    e conceitos de justiça de classificação que podem ser aplicados.
    """
    logging.info("Iniciando análise de justiça (fairness) para o modelo PyTorch...")

    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    unique_classes = np.unique(all_targets)
    logging.info(f"Classes únicas no dataset: {unique_classes}")

    print("\n--- Análise de Fairness (Performance por Classe) ---")
    print(f"Dataset: {dataset_name}")

    # Análise de performance por classe
    overall_accuracy = accuracy_score(all_targets, all_preds)
    print(f"Acurácia Geral: {overall_accuracy:.4f}\n")

    metrics_per_class = {}
    for cls in unique_classes:
        class_indices = (all_targets == cls)
        preds_for_class = all_preds[class_indices]
        targets_for_class = all_targets[class_indices]

        if len(targets_for_class) > 0:
            acc = accuracy_score(targets_for_class, preds_for_class)
            prec = precision_score(targets_for_class, preds_for_class, average='binary', pos_label=cls, zero_division=0)
            rec = recall_score(targets_for_class, preds_for_class, average='binary', pos_label=cls, zero_division=0)
            f1 = f1_score(targets_for_class, preds_for_class, average='binary', pos_label=cls, zero_division=0)
            metrics_per_class[cls] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
            print(f"Classe {cls}:")
            print(f"  Acurácia: {acc:.4f}")
            print(f"  Precisão: {prec:.4f}")
            print(f"  Recall: {rec:.4f}")
            print(f"  F1-Score: {f1:.4f}\n")
        else:
            logging.warning(f"Nenhuma amostra para a classe {cls} no conjunto de teste.")

    print("--------------------------------------------------")
    logging.info("Análise de justiça finalizada.")

    # Exemplo conceitual de como AIF360/Fairlearn seriam usados com atributos protegidos (se existissem)
    # if dataset_name == 'simulated_protected_data': # Exemplo hipotético
    #     # Supondo que você tenha atributos protegidos e labels binários
    #     # O modelo PyTorch precisaria ser integrado a esta estrutura
    #     data_dict = {
    #         'features': pd.DataFrame(all_preds), # Usando predições como features de exemplo
    #         'labels': pd.Series(all_targets),
    #         'protected_attributes': pd.DataFrame({'group': np.random.randint(0, 2, len(all_targets))}),
    #     }
    #     bld = BinaryLabelDataset(df=pd.DataFrame(data_dict), label_names=['labels'], 
    #                              protected_attribute_names=['group'],
    #                              unprivileged_groups=[{'group': 0}], privileged_groups=[{'group': 1}])
    #     
    #     metric = ClassificationMetric(bld, bld, 
    #                                   unprivileged_groups=[{'group': 0}], 
    #                                   privileged_groups=[{'group': 1}])
    #     
    #     print(f"\nStatistical Parity Difference (AIF360): {metric.statistical_parity_difference():.4f}")
    #     print(f"Equal Opportunity Difference (AIF360): {metric.equal_opportunity_difference():.4f}")
    #     
    #     # Fairlearn example
    #     gm = MetricFrame(metrics=accuracy_score, 
    #                      y_true=all_targets, y_pred=all_preds, 
    #                      sensitive_features=data_dict['protected_attributes']['group'])
    #     print(f"\nDemographic Parity Difference (Fairlearn): {demographic_parity_difference(y_true=all_targets, y_pred=all_preds, sensitive_features=data_dict['protected_attributes']['group']):.4f}")
    #     print(f"Equalized Odds Difference (Fairlearn): {equalized_odds_difference(y_true=all_targets, y_pred=all_preds, sensitive_features=data_dict['protected_attributes']['group']):.4f}") 