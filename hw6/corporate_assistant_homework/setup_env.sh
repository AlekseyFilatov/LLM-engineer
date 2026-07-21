#!/bin/bash
echo "📦 [SETUP] Накатываем промышленное ИИ-окружение в ваше виртуальное окружение..."

# 1. Универсальная проверка: активировано ли вообще venv?
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️ Виртуальное окружение не активировано в терминале!"
    if [ -d ".venv_wsl" ]; then
        echo "🔄 Найдена папка .venv_wsl. Активируем..."
        source .venv_wsl/bin/activate
    elif [ -d "ml_env" ]; then
        echo "🔄 Найдена папка ml_env. Активируем..."
        source ml_env/bin/activate
    else
        echo "❌ Ошибка: Локальное окружение не найдено. Создайте его через: python3 -m venv .venv_wsl"
        exit 1
    fi
else
    echo "✅ Зафиксировано активное окружение: $(basename "$VIRTUAL_ENV")"
fi

echo "🚀 Установщик pip обновляется..."
pip install --upgrade pip -q

echo "📥 Установка полного стека библиотек (RAG + LangGraph + Transformers)..."
pip install \
    pdfplumber \
    beautifulsoup4 \
    chromadb \
    transformers \
    dspy-ai \
    aiohttp \
    pyjwt \
    pyyaml \
    requests \
    accelerate \
    sentence-transformers \
    rank_bm25 \
    python-dotenv \
    langgraph \
    langchain-experimental \
    nltk \
    pydantic \
    -q

echo "✅ [SUCCESS] Все библиотеки успешно установлены!"
echo "编 На всякий случай фиксируем их в requirements.txt..."
pip freeze > requirements.txt
#pip cache purge