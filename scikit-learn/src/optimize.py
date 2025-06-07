from sklearn.model_selection import GridSearchCV
from model import build_pipeline
from data import load_iris_data
import logging

def optimize_hyperparameters(X_train, y_train, param_grid=None, cv=5):
    """Executa GridSearchCV para otimizar hiperparâmetros do RandomForest."""
    if param_grid is None:
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 5, 10],
            'clf__min_samples_split': [2, 5]
        }
    pipeline = build_pipeline()
    grid = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    logging.info(f'Melhores parâmetros: {grid.best_params_}')
    logging.info(f'Melhor acurácia: {grid.best_score_:.4f}')
    return grid.best_estimator_, grid.best_params_, grid.best_score_ 