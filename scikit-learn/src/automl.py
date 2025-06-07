from tpot import TPOTClassifier
import logging
import joblib

def run_automl(X_train, y_train, X_test, y_test, generations=5, population_size=20, cv=5, random_state=42, n_jobs=-1, export_path='tpot_best_pipeline.py', model_path='tpot_pipeline.joblib'):
    """Executa AutoML com TPOT, salva o melhor pipeline e gera código explicativo."""
    logging.info('Iniciando AutoML com TPOT...')
    tpot = TPOTClassifier(generations=generations, population_size=population_size, cv=cv, verbosity=2, random_state=random_state, n_jobs=n_jobs)
    tpot.fit(X_train, y_train)
    score = tpot.score(X_test, y_test)
    logging.info(f'Acurácia do pipeline TPOT no teste: {score:.4f}')
    tpot.export(export_path)
    logging.info(f'Código do pipeline ótimo salvo em: {export_path}')
    joblib.dump(tpot.fitted_pipeline_, model_path)
    logging.info(f'Pipeline treinado salvo em: {model_path}')
    return score, export_path, model_path 