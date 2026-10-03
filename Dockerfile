FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEMO_BACKEND=openrouter \
    DEMO_MODEL=stealth/space-bunny-alpha \
    RANK_MODEL=stealth/space-bunny-alpha

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY src ./src
COPY demo ./demo
COPY data/study_products.jsonl data/panels.json data/product_images.json data/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.demo_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
