FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar pyproject.toml y código fuente
COPY pyproject.toml .
COPY rag_app ./rag_app

# Instalar dependencias desde pyproject.toml
RUN pip install --no-cache-dir .

# Copiar datos
COPY data ./data

EXPOSE 8501

CMD ["streamlit", "run", "rag_app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]