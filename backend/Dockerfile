FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    CACHE_DIR=/app/cache \
    DOCS_DIR=/app/docs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appgroup \
    && useradd \
        --uid 10001 \
        --gid appgroup \
        --create-home \
        --no-log-init \
        appuser

COPY backend/requirements-cpu.txt /tmp/requirements-cpu.txt
COPY backend/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements-cpu.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=appuser:appgroup backend/ /app/

RUN mkdir -p /app/cache /home/appuser/.cache/huggingface \
    && chown -R appuser:appgroup /app/cache /home/appuser/.cache

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]