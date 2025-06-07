import openai
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def generate_nlg_report(accuracy: float, dataset_name: str, model_type: str) -> str:
    """Gera um relatório em linguagem natural sobre o desempenho do modelo usando OpenAI."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("Variável de ambiente OPENAI_API_KEY não definida. Não é possível gerar relatório NLG.")
        return "Erro: OPENAI_API_KEY não configurada."

    # Certifique-se de que a chave API está configurada
    openai.api_key = openai_api_key

    prompt = f"""Gere um relatório conciso sobre o desempenho de um modelo de {model_type} treinado no dataset {dataset_name}. A acurácia final alcançada foi de {accuracy:.2f}%. Inclua uma análise sucinta sobre o que essa acurácia significa e possíveis próximos passos para melhoria. O relatório deve ser profissional e ter no máximo 200 palavras."""

    try:
        logging.info("Gerando relatório NLG com OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um analista de IA especializado em explicar o desempenho de modelos de Machine Learning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        report = response.choices[0].message.content.strip()
        logging.info("Relatório NLG gerado com sucesso.")
        return report
    except openai.APIError as e:
        logging.error(f"Erro na API da OpenAI: {e}")
        return f"Erro ao gerar relatório NLG: {e}"
    except Exception as e:
        logging.error(f"Erro inesperado ao gerar relatório NLG: {e}")
        return f"Erro inesperado ao gerar relatório NLG: {e}" 