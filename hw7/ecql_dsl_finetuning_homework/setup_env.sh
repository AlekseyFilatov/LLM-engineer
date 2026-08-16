#!/usr/bin/env bash
set -euo pipefail

echo "📦 [SETUP] Накатываем промышленное ИИ-окружение на Ubuntu с поддержкой GPU..."

# 1. Определение окружения
ENV_DIR=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "✅ Зафиксировано active окружение в терминале: $(basename "$VIRTUAL_ENV")"
    ENV_DIR="$VIRTUAL_ENV"
elif [ -d ".venv" ]; then
    echo "🔄 Найдена папка .venv. Используем её..."
    ENV_DIR="$(pwd)/.venv"
elif [ -d "ml_env" ]; then
    echo "🔄 Найдена папка ml_env. Используем её..."
    ENV_DIR="$(pwd)/ml_env"
else
    echo "🔄 Локальное окружение не найдено. Создаем чистый .venv..."
    if ! python3 -m venv .venv 2>/dev/null; then
        echo "❌ Ошибка: В системе отсутствует пакет python3-venv."
        echo "💡 Выполните: sudo apt update && sudo apt install -y python3-venv"
        exit 1
    fi
    ENV_DIR="$(pwd)/.venv"
fi

# Путь к pip внутри окружения
ENV_PIP="$ENV_DIR/bin/pip"

if [ ! -f "$ENV_PIP" ]; then
    echo "❌ Не удалось найти pip в окружении по пути: $ENV_PIP"
    exit 1
fi

echo "🚀 Обновляем pip..."
"$ENV_PIP" install --upgrade pip --timeout 100 --retries 5 -q

# === КРИТИЧЕСКОЕ ДОПОЛНЕНИЕ ДЛЯ GPU (CUDA 12) ===
echo "⚡ Первичная установка PyTorch с поддержкой CUDA 12.1/12.4..."
# Скачиваем нативные бинарники PyTorch с поддержкой CUDA напрямую с официального сервера хранения
"$ENV_PIP" install torch torchvision torchaudio \
    --index-url https://pytorch.org \
    --timeout 1000 --retries 10 -q

echo "📥 Установка полного стека библиотек (RAG + LangGraph + Fine-Tuning Core)..."
# Стек ставится поверх установленного CUDA Torch
"$ENV_PIP" install \
    pdfplumber \
    beautifulsoup4 \
    chromadb \
    transformers \
    dspy-ai \
    aiohttp \
    pyjwt \
    pyyaml \
    peft \
    requests \
    accelerate \
    langchain-openai \
    sentence-transformers \
    langchain-huggingface \
    rank_bm25 \
    python-dotenv \
    langgraph \
    langchain-experimental \
    nltk \
    pydantic \
    trl \
    bitsandbytes \
    scikit-learn \
    packaging \
    --timeout 1000 \
    --retries 10 \
    -q

echo "✅ [SUCCESS] Все библиотеки успешно установлены!"

echo "📝 Фиксация зависимостей в локальный requirements.txt..."
"$ENV_PIP" freeze --local > requirements.txt

echo "🧹 Очистка временного кэша pip..."
"$ENV_PIP" cache purge -q
echo "✅ Кэш pip очищен."

# Создаем внутри проекта папку для кэша моделей ИИ
mkdir -p "$(pwd)/src/storage/hf_cache"
mkdir -p "$(pwd)/src/storage/nltk_data"

echo -e "\n💡 Чтобы начать работу, активируйте окружение командой:"
echo -e "   \033[1;32mset +u && source $ENV_DIR/bin/activate && set -u\033[0m"

echo -e "\n📦 Рекомендуется добавить в ваш .env файл или экспортировать пути для кэша моделей:"
echo -e "   \033[1;36mexport HF_HOME=\"$(pwd)/src/storage/hf_cache\"\033[0m"
echo -e "   \033[1;36mexport NLTK_DATA=\"$(pwd)/src/storage/nltk_data\"\033[0m\n"

# Проверяем, видит ли Python вашу видеокарту прямо сейчас
echo "🔍 Проверка доступности графического процессора (GPU)..."
"$ENV_DIR/bin/python" -c "
import torch
if torch.cuda.is_available():
    print(f'   \033[1;32m✅ CUDA доступна! Обнаружена видеокарта: {torch.cuda.get_device_name(0)}\033[0m')
    print(f'   Текущая версия CUDA в PyTorch: {torch.version.cuda}')
else:
    print('   \033[1;31m⚠️ CUDA недоступна. Скрипт будет обучаться на CPU (очень медленно).\033[0m')
    print('   💡 Убедитесь, что установлены официальные драйверы NVIDIA.')
"
