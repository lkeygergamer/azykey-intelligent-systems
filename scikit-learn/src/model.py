from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any
import numpy as np

def build_pipeline(n_estimators=100, data_type='tabular', online_mode=False) -> Pipeline:
    """Cria um pipeline de ML multimodal, com opção para aprendizado online."""
    if online_mode:
        if data_type == 'tabular':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SGDClassifier(random_state=42, tol=1e-3)) # tol added for convergence
            ])
        elif data_type == 'text':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SGDClassifier(random_state=42, tol=1e-3)) # Use SGDClassifier for text
            ])
        elif data_type == 'image':
            pipeline = Pipeline([
                ('flatten', FunctionTransformer(lambda x: np.array([i.flatten() for i in x]), validate=False)),
                ('clf', SGDClassifier(random_state=42, tol=1e-3)) # Use SGDClassifier for image
            ])
        else:
            raise ValueError(f"Tipo de dado não suportado para aprendizado online: {data_type}")
    else: # Existing offline mode
        if data_type == 'tabular':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=42))
            ])
        elif data_type == 'text':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ])
        elif data_type == 'image':
            pipeline = Pipeline([
                ('flatten', FunctionTransformer(lambda x: np.array([i.flatten() for i in x]), validate=False)),
                ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=42))
            ])
        else:
            raise ValueError(f"Tipo de dado não suportado: {data_type}")
    return pipeline 