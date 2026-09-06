#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[0;32m📦 [TRITON CLIENT SETUP] Развертывание окружения ИИ-конвейера...\033[0m"

ENV_DIR=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "✅ Зафиксировано активное окружение в терминале: $(basename "$VIRTUAL_ENV")"
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

ENV_PIP="$ENV_DIR/bin/pip"
ENV_PYTHON="$ENV_DIR/bin/python"

echo "🔄 Обновление менеджера пакетов pip..."
"$ENV_PIP" install --upgrade pip -q

echo "📥 Шаг 1/2: Проверка вычислительного бэкенда..."

# Проверяем, установлен ли уже torch глобально или в окружении (например, внутри контейнера Triton)
if "$ENV_PYTHON" -c "import torch" &>/dev/null; then
    echo "✅ PyTorch уже предустановлен в этой среде. Пропускаем тяжелую загрузку."
else
    # Если мы не в контейнере, проверяем видеокарту
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️ Предупреждение: nvidia-smi не найден. Установка PyTorch (CPU-версия)..."
        "$ENV_PIP" install torch torchvision --index-url https://pytorch.org -q
    else
        echo "✅ Обнаружен GPU. Установка PyTorch с поддержкой CUDA..."
        # Добавлен флаг --trusted-host на случай проблем с SSL-сертификатами в WSL2/Docker
        # Также в качестве запасного варианта pip попробует стандартный PyPI, если whl-зеркало недоступно
        "$ENV_PIP" install torch torchvision \
            --index-url https://pytorch.org \
            --trusted-host download.pytorch.org -q || {
                echo "⚠️ Не удалось скачать с официального зеркала PyTorch. Пробуем стандартный PyPI..."
                "$ENV_PIP" install torch torchvision -q
            }
    fi
fi

echo "📥 Шаг 2/2: Синхронизация зависимостей Triton Client пайплайна..."
"$ENV_PIP" install \
    "tritonclient[http]>=2.54.0" \
    "numpy>=1.24.0,<2.0.0" \
    "transformers>=4.45.0" \
    tqdm \
    python-dotenv \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org -q

echo "✅ [SUCCESS] Все библиотеки успешно установлены!"

echo "📝 Фиксация зависимостей в локальный requirements.txt..."
"$ENV_PIP" freeze --local > requirements.txt

echo "🧹 Очистка временного кэша pip..."
"$ENV_PIP" cache purge -q
echo "✅ Кэш pip очищен."
