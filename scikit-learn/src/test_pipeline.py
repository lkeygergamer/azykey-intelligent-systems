import pytest
from data import load_iris_data
from model import build_pipeline
from train import train_pipeline
from evaluate import evaluate_pipeline
from sklearn.datasets import load_iris

@pytest.fixture
def iris_data():
    return load_iris_data()

@pytest.fixture
def pipeline():
    return build_pipeline()

def test_data_shape(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_pipeline_training(pipeline, iris_data):
    X_train, X_test, y_train, y_test = iris_data
    trained = train_pipeline(pipeline, X_train, y_train)
    assert hasattr(trained, 'predict')

def test_pipeline_evaluation(pipeline, iris_data):
    X_train, X_test, y_train, y_test = iris_data
    trained = train_pipeline(pipeline, X_train, y_train)
    iris = load_iris()
    report = evaluate_pipeline(trained, X_test, y_test, target_names=iris.target_names)
    assert 'precision' in report 