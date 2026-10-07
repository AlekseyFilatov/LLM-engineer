FROM docker.io/library/ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app

RUN useradd -r -s /bin/false appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Копируем только pyproject.toml
COPY pyproject.toml ./

# Ставим пакеты напрямую из .toml без использования потенциально битого кэша
RUN uv pip install --system --break-system-packages -r pyproject.toml

RUN python3.12 -m spacy download ru_core_news_md --break-system-packages

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser ui_client.py index_catalog.py ./

RUN rm pyproject.toml

USER appuser

EXPOSE 7860

CMD ["python3.12", "ui_client.py"]
