FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY src/config.yaml ./src/config.yaml
COPY src/report_template.html ./src/report_template.html

CMD ["python", "src/main.py"] 