# FastH3 on a Hugging Face Inference Endpoint (H200, custom container).
# Base: FastVideo's CUDA 13 dev image (has torch cu130 + fastvideo installed).
FROM ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest

RUN /opt/venv/bin/pip install -q fastapi uvicorn pydantic

WORKDIR /app
COPY server.py /app/server.py

# HF endpoints route traffic to port 80 and probe /health.
ENV HF_HOME=/repository/hf-cache
EXPOSE 80
CMD ["/opt/venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
