from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_html_report(report: str, cm, target_names, output_path='report.html'):
    """Gera um relatório HTML com o relatório de classificação e matriz de confusão."""
    # Salvar matriz de confusão como imagem
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    img_path = 'confusion_matrix.png'
    plt.savefig(img_path)
    plt.close()

    # Carregar template Jinja2
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('report_template.html')
    html_content = template.render(report=report, cm_image=img_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Relatório HTML gerado em: {output_path}') 