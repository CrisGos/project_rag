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

### Ejecución
Para correr la aplicación (ahora migrada a Chainlit):
```bash
uv run chainlit run rag_app/ui/chat.py
```

### Docker
El contenedor actualizado utiliza `uv` internamente.
```bash
docker build -t rag_app .
docker run -p 8501:8501 rag_app
```
