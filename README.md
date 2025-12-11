# project_rag
Project using a local RAG with langchain

## Gestión de Dependencias (uv)
Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para la gestión de dependencias y entornos virtuales.

### Instalación
1. Instalar uv:
    ```bash
    pip install uv
    # o
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Sincronizar entorno:
    ```bash
    uv sync
    ```

### Servicios Requeridos (Docker)
Esta aplicación requiere una base de datos vectorial (Weaviate) y un servidor de modelos (Ollama).
1. Asegúrate de tener **Docker Desktop** instalado y **corriendo**.
2. Levanta los servicios:
    ```bash
    docker-compose -f docker-compose.weaviate.yml up -d
    ```

### Ejecución
Una vez que los servicios de Docker estén corriendo:
```bash
uv run chainlit run rag_app/ui/chat.py
```

### Docker
El contenedor actualizado utiliza `uv` internamente.
```bash
docker build -t rag_app .
docker run -p 8000:8000 rag_app
```
