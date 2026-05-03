# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS builder
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --user -r requirements.txt

COPY app ./app
# Bake the trained model into the image so deployments are self-contained.
RUN PYTHONPATH=/app python -m app.ml.train


FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    PATH="/home/app/.local/bin:${PATH}" \
    PYTHONPATH=/app

RUN useradd -ms /bin/bash app
USER app

COPY --from=builder --chown=app:app /root/.local /home/app/.local
COPY --from=builder --chown=app:app /app/app /app/app
COPY --from=builder --chown=app:app /app/models /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)" || exit 1

CMD ["python", "-m", "app.main"]
