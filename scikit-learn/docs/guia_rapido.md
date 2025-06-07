# Guia Rápido

## Instalação

```bash
pip install -r requirements.txt
```

## Execução básica

```bash
python src/main.py --report --plot-cm
```

## Usando Docker

```bash
docker build -t sklearn-pipeline .
docker run --rm -v $(pwd)/src:/app/src sklearn-pipeline --report --plot-cm
``` 