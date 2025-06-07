import openai
import os
import logging
import tiktoken
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_llm_response(prompt: str, api_key: str = None, model: str = "gpt-3.5-turbo", max_tokens: int = 500, temperature: float = 0.7) -> str:
    """Obtém uma resposta de texto de um LLM da OpenAI."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("Chave da API OpenAI não fornecida. Incapaz de gerar texto.")
        return ""
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erro ao chamar a API OpenAI: {e}")
        return ""

def generate_synthetic_text_data(texts: List[str], labels: List[int], num_samples_per_class: int = 5, api_key: str = None, model: str = "gpt-3.5-turbo") -> Tuple[List[str], List[int]]:
    """Gera dados de texto sintéticos para aumento de dados usando um LLM."""
    logging.info(f'Iniciando geração de dados sintéticos com LLM (Modelo: {model})... ')
    synthetic_texts = []
    synthetic_labels = []

    unique_labels = sorted(list(set(labels)))
    for label in unique_labels:
        # Obter textos de exemplo para a classe atual
        class_texts = [texts[i] for i, l in enumerate(labels) if l == label]
        if not class_texts:
            logging.warning(f"Nenhum texto encontrado para a classe {label}. Pulando geração para esta classe.")
            continue

        # Criar um prompt para gerar textos semelhantes
        example_texts = "\n".join(class_texts[:min(3, len(class_texts))]) # Usar até 3 exemplos
        prompt = f"""
        Gere {num_samples_per_class} textos curtos e variados que são semelhantes no estilo e conteúdo aos exemplos fornecidos. Não inclua numeração ou marcadores de lista. Cada texto deve estar em uma nova linha. O tema geral é sobre {labels[texts.index(class_texts[0])]}.\n\nExemplos:\n{example_texts}
        """
        
        generated_content = get_llm_response(prompt, api_key=api_key, model=model)
        
        if generated_content:
            new_texts = [t.strip() for t in generated_content.split('\n') if t.strip()]
            synthetic_texts.extend(new_texts)
            synthetic_labels.extend([label] * len(new_texts))
            logging.info(f"Geradas {len(new_texts)} amostras sintéticas para a classe {label}.")
        else:
            logging.warning(f"Nenhum texto gerado para a classe {label}.")

    return synthetic_texts, synthetic_labels

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Conta o número de tokens em um texto para um dado modelo."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Erro ao contar tokens para o modelo {model_name}: {e}. Retornando 0.")
        return 0 