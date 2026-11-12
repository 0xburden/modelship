FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install --no-cache-dir diffusers==0.35.1 transformers==4.56.2 accelerate==1.10.1 safetensors==0.6.2 fastapi==0.116.1 uvicorn==0.35.0 pillow==11.3.0 sentencepiece==0.2.1 protobuf==6.32.1 pydantic==2.11.7
RUN useradd --uid 10001 --create-home app && mkdir -p /cache/huggingface && chown -R app:app /cache
WORKDIR /app
COPY server.py /app/server.py
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
