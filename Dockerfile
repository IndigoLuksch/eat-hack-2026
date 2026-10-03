FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEMO_BACKEND=trl

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-demo.txt .
# PEFT is how a trl adapter is loaded. CPU torch stays inside a 16 GB instance.
RUN pip install --no-cache-dir -r requirements-demo.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers>=4.51.0" "peft>=0.15.0" accelerate

COPY src ./src
COPY demo ./demo
COPY data/study_products.jsonl data/panels.json data/product_images.json data/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.demo_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
