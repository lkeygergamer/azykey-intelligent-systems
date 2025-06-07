from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from typing import Any, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def cross_validate_pipeline(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> float:
    """Executa validação cruzada e retorna a acurácia média."""
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
    return scores.mean()

def train_pipeline(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray, online_mode: bool = False, all_seen_classes: np.ndarray = None) -> Pipeline:
    """Treina o pipeline no conjunto de treino, com opção para aprendizado online."""
    if online_mode:
        logging.info('Treinando pipeline em modo online (partial_fit)...')
        try:
            # Acessa o estimador final do pipeline
            clf = pipeline.named_steps['clf']
            
            # Se o estimador suporta partial_fit
            if hasattr(clf, 'partial_fit'):
                # Garante que as classes sejam inicializadas para SGDClassifier, etc.
                if not hasattr(clf, 'classes_') or clf.classes_ is None:
                    if all_seen_classes is None:
                        logging.warning("Para aprendizado online, é recomendável fornecer todas as classes possíveis ao inicializar o modelo. Usando classes de y_train.")
                        initial_classes = np.unique(y_train)
                    else:
                        initial_classes = all_seen_classes
                    clf.partial_fit(X_train, y_train, classes=initial_classes)
                else:
                    clf.partial_fit(X_train, y_train)
            else:
                logging.warning("Modelo não suporta partial_fit para aprendizado online. Usando fit() completo.")
                pipeline.fit(X_train, y_train) # Fallback para fit completo
        except Exception as e:
            logging.error(f"Erro ao treinar em modo online: {e}. Revertendo para fit() completo.")
            pipeline.fit(X_train, y_train) # Fallback para fit completo
    else:
        logging.info('Treinando pipeline em modo offline (fit completo)...')
        pipeline.fit(X_train, y_train)
    return pipeline 