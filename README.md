# project_rag
Project using a local RAG with langchain + Chainlit UI.

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
    # Si tienes TAVILY_API_KEY configurada en tu entorno, docker-compose la tomará automáticamente.
    docker-compose -f docker-compose.weaviate.yml up -d
    ```

### Ejecución Local
Una vez que los servicios de Docker estén corriendo (Weaviate y Ollama):
```bash
# Ejecutar la UI de Chainlit
uv run chainlit run rag_app/ui/chat.py --host 0.0.0.0 --port 8000
```
Accede a http://localhost:8000.

### Agente Tavily (Opcional)
El sistema incluye un agente de respaldo que utiliza Tavily Search cuando la información no se encuentra en los documentos indexados.

1.  **Configuración**:
    Obtén tu API Key en [tavily.com](https://tavily.com/).
    
    Linux/Mac:
    ```bash
    export TAVILY_API_KEY="tvly-..."
    ```
    Windows (PowerShell):
    ```powershell
    $env:TAVILY_API_KEY="tvly-..."
    ```

2.  **Uso**:
    El agente se activará automáticamente si el `check_document_exists` determina que el documento solicitado no está en la base de conocimientos.

### Docker (Full Stack)
Para ejecutar toda la aplicación (incluida la UI) en Docker:

1. Build:
    ```bash
    docker build -t rag_app .
    ```

2. Run (con variables de entorno):
    ```bash
    # Básico
    docker run -p 8000:8000 --network project_rag_default rag_app

    # Con Tavily (pasando la variable)
    docker run -p 8000:8000 --network project_rag_default -e TAVILY_API_KEY=$TAVILY_API_KEY rag_app
    ```
    *Nota: Asegúrate de que el contenedor esté en la misma red que Weaviate y Ollama si los corriste con compose (`project_rag_default` es el nombre común, verifica con `docker network ls`).*
    
    Alternativamente, usa simplemente `docker-compose`:
    ```bash
    docker-compose -f docker-compose.weaviate.yml up -d --build
    ```

