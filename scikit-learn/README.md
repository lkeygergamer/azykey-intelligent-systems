# scikit-learn

Esta pasta contém projetos, exemplos e código-fonte utilizando a biblioteca scikit-learn para machine learning.

## Estrutura
- `src/`: Código-fonte principal dos projetos com scikit-learn.
- `examples/`: Exemplos práticos de uso do scikit-learn.
- `requirements.txt`: Dependências necessárias para rodar os projetos.
- `Dockerfile`: Imagem para execução e deploy do pipeline.

## Como começar
1. Instale as dependências: `pip install -r requirements.txt`
2. Explore os exemplos na pasta `examples/` ou desenvolva seus próprios projetos em `src/`.

## Executando com Docker

```bash
docker build -t sklearn-pipeline .
docker run --rm -v $(pwd)/src:/app/src sklearn-pipeline --report --plot-cm
```

Você pode passar outros argumentos do script normalmente após o nome da imagem. 