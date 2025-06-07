import openai
import os
import logging

def generate_nlg_report(report_text, cm, target_names, api_key=None, model="gpt-3.5-turbo"):
    """Gera um relatório em linguagem natural explicando os resultados do modelo."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("Chave da API OpenAI não fornecida. NLG desativado.")
        return None
    openai.api_key = api_key
    prompt = f"""
Você é um especialista em ciência de dados. Analise o seguinte relatório de classificação e matriz de confusão, explique os resultados, destaque pontos fortes e fracos do modelo e sugira próximos passos.\n\nRelatório:\n{report_text}\n\nMatriz de confusão:\n{cm}\n\nClasses: {target_names}\n"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3
    )
    nlg_report = response.choices[0].message.content.strip()
    return nlg_report 