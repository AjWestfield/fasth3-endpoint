# FastH3 on a Hugging Face Inference Endpoint (H200, custom container).
FROM ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest

# Match the proven RunPod setup: update FastVideo to head (the baked version
# predates the FastH3 checkpoint's VSA gate weights and fails to load it).
RUN cd /FastVideo && git pull && /opt/venv/bin/pip install -q -e ".[dev]" && \
    /opt/venv/bin/pip install -q fastapi uvicorn pydantic

WORKDIR /app
COPY server.py /app/server.py

ENV HF_HOME=/repository/hf-cache
EXPOSE 80
CMD ["/opt/venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
