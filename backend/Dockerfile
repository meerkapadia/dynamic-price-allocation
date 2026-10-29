FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PRICING_DB_PATH=/data/store.db OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pricing_engine ./pricing_engine
COPY api ./api
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0)"
# one process: pricing runs use a thread pool (PRICING_WORKERS); approvals are serialised by SQLite
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
