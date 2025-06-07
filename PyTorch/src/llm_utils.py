import openai
import os
import logging
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Conta o número de tokens em um texto usando tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        logging.warning(f"Modelo {model} não encontrado para tokenização, usando cl100k_base.")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def generate_synthetic_text(prompt: str, max_tokens: int = 100) -> str:
    """Gera texto sintético usando a API da OpenAI."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("Variável de ambiente OPENAI_API_KEY não definida. Não é possível gerar texto sintético.")
        return "Erro: OPENAI_API_KEY não configurada."
    
    openai.api_key = openai_api_key

    try:
        logging.info("Gerando texto sintético com OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente criativo que pode gerar descrições textuais ricas para objetos e conceitos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        generated_text = response.choices[0].message.content.strip()
        logging.info("Texto sintético gerado com sucesso.")
        return generated_text
    except openai.APIError as e:
        logging.error(f"Erro na API da OpenAI: {e}")
        return f"Erro ao gerar texto sintético: {e}"
    except Exception as e:
        logging.error(f"Erro inesperado ao gerar texto sintético: {e}")
        return f"Erro inesperado ao gerar texto sintético: {e}"

def augment_data_with_llm(dataset_name: str, num_samples: int = 1) -> None:
    """Simula o aumento de dados com texto sintético gerado por LLM.
    Para datasets de imagem, isso pode significar gerar descrições ricas para classes ou exemplos.
    """
    logging.info(f"Simulando aumento de dados para {dataset_name} com texto sintético...")
    
    if dataset_name == 'mnist':
        class_descriptions = {
            0: "representa o número zero, frequentemente retratado como um círculo ou oval.",
            1: "representa o número um, geralmente um traço vertical simples.",
            2: "representa o número dois, com uma curva na parte superior e uma linha horizontal na base.",
            3: "representa o número três, com duas curvas para a direita, empilhadas.",
            4: "representa o número quatro, com uma linha vertical, uma horizontal e outra vertical curta.",
            5: "representa o número cinco, com um traço horizontal, um vertical e uma curva inferior.",
            6: "representa o número seis, com uma curva na parte superior que termina em um círculo fechado na base.",
            7: "representa o número sete, com um traço horizontal superior e uma linha diagonal descendente.",
            8: "representa o número oito, composto por dois círculos empilhados ou uma única linha contínua que os forma.",
            9: "representa o número nove, com um círculo na parte superior e uma linha vertical descendente."
        }
        for i in range(num_samples):
            for digit, desc in class_descriptions.items():
                prompt = f"Descreva detalhadamente uma imagem do dígito {digit}, considerando variações de estilo de escrita manual. Foco na forma e características visuais. O texto deve ter no máximo 50 palavras." 
                synthetic_text = generate_synthetic_text(prompt, max_tokens=50)
                logging.info(f"Texto sintético para o dígito {digit}: {synthetic_text}")
                print(f"\n--- Descrição para o Dígito {digit} ({i+1}/{num_samples}) ---")
                print(synthetic_text)
                print("--------------------------------------------------")
    elif dataset_name == 'cifar10':
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        for i in range(num_samples):
            for cls in cifar10_classes:
                prompt = f"Gere uma descrição viva e concisa de uma imagem de um(a) {cls}. Inclua detalhes sobre cores típicas, ambiente e características distintivas do objeto. O texto deve ter no máximo 50 palavras."
                synthetic_text = generate_synthetic_text(prompt, max_tokens=50)
                logging.info(f"Texto sintético para a classe {cls}: {synthetic_text}")
                print(f"\n--- Descrição para {cls} ({i+1}/{num_samples}) ---")
                print(synthetic_text)
                print("--------------------------------------------------")
    else:
        logging.warning(f"Aumento de dados textual não implementado para o dataset: {dataset_name}")
        print(f"Aumento de dados textual não implementado para o dataset: {dataset_name}")

    logging.info("Aumento de dados com texto sintético finalizado.") 