FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache
ENV PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY . .

# Fixed knowledge-base demo: use the locally built vector store copied into the image.
RUN test -s vector_store.json && test -s .ingest_state.json && test -d .hf_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=5)"

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
