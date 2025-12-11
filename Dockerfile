FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# System deps for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY rag_app ./rag_app
COPY .streamlit ./.streamlit

# Install the project itself
RUN uv sync --frozen --no-dev

# Create data directories
RUN mkdir -p ./data/logs ./data/pdfs

EXPOSE 8000

# Run chainlit
ENV PATH="/app/.venv/bin:$PATH"
CMD ["chainlit", "run", "rag_app/ui/chat.py", "--host", "0.0.0.0", "--port", "8000"]