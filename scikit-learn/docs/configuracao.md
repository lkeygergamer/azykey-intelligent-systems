# Configuração via YAML

O arquivo `src/config.yaml` permite definir hiperparâmetros, caminhos e opções do pipeline de forma centralizada.

Exemplo:

```yaml
cv: 5
model_path: iris_pipeline.joblib
plot_cm: true
test_size: 0.2
random_state: 42
n_estimators: 100
```

Você pode sobrescrever qualquer valor via argumentos de linha de comando. 