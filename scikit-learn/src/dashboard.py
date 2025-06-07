import streamlit as st
import pandas as pd
import joblib
from data import load_dataset
from model import build_pipeline
from train import train_pipeline
from evaluate import evaluate_pipeline, plot_confusion_matrix
from report import generate_html_report
import os

st.set_page_config(page_title="IA Pipeline Dashboard", layout="wide")
st.title("IA Pipeline Dashboard (scikit-learn)")

# Upload de dados
st.sidebar.header("Upload de Dados")
dataset_option = st.sidebar.selectbox("Escolha o dataset", ("iris", "wine", "digits"))

if st.sidebar.button("Carregar e Treinar Pipeline"):
    X_train, X_test, y_train, y_test, target_names = load_dataset(name=dataset_option)
    pipeline = build_pipeline()
    pipeline = train_pipeline(pipeline, X_train, y_train)
    st.success("Pipeline treinado!")

    # Avaliação
    report = evaluate_pipeline(pipeline, X_test, y_test, target_names=target_names)
    st.subheader("Relatório de Classificação")
    st.text(report)

    # Matriz de confusão
    st.subheader("Matriz de Confusão")
    plot_confusion_matrix(pipeline, X_test, y_test, target_names=target_names)
    st.pyplot()

    # Download do modelo
    model_path = f"pipeline_{dataset_option}.joblib"
    joblib.dump(pipeline, model_path)
    with open(model_path, "rb") as f:
        st.download_button("Baixar modelo treinado", f, file_name=model_path)

    # Geração de relatório HTML
    from sklearn.metrics import confusion_matrix
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    generate_html_report(report, cm, target_names, output_path="report_dashboard.html")
    with open("report_dashboard.html", "rb") as f:
        st.download_button("Baixar relatório HTML", f, file_name="report_dashboard.html")

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido com Streamlit e scikit-learn. IA de ponta para todos!") 