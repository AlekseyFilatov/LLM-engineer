# --- ЭТАП 1: Сборка зависимостей (Builder) ---
FROM ubuntu:24.04 AS builder
WORKDIR /app

# Копируем pyproject.toml заранее для эффективного кэширования
COPY pyproject.toml ./

# Ставим только необходимый минимум для компиляции Python-пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Забираем чистый бинарник uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Синхронизируем зависимости (uv создаст /app/.venv на базе Ubuntu 24.04)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --no-editable

RUN uv pip install "https://github.com/explosion/spacy-models/releases/download/sv_core_news_md-3.7.0/sv_core_news_md-3.7.0-py3-none-any.whl"

# --- ЭТАП 2: Ультра-легковесный Рантайм на базе чистой Ubuntu 24.04 ---
FROM ubuntu:24.04
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Создаём непривилегированного пользователя БЕЗ домашней папки
RUN useradd -r appuser

# Минимальный набор библиотек: python3.12, libgomp1 (для ML) и ca-certificates
# Команда rm -rf в конце строки полностью вычищает кэш apt, делая образ "slim"
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.12 /usr/bin/python

# Копируем всё сразу с правами appuser (исключает раздувание образа слоями chown)
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser ui_client.py index_catalog.py pyproject.toml ./

USER appuser

EXPOSE 7860
CMD ["/app/.venv/bin/python", "ui_client.py"]