#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[0;32m📦 [OMNI SETUP v2] Развертывание динамического мультимодального окружения E-Corp...\033[0m"

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

ENV_PIP="$ENV_DIR/bin/pip"
echo "🔄 Обновление менеджера пакетов pip..."
"$ENV_PIP" install --upgrade pip -q


if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ Предупреждение: nvidia-smi не найден. Возможно, CUDA не установлена. Будет установлена CPU-версия."
    "$ENV_PIP" install torch torchvision torchaudio -q
else
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1 | cut -d'.' -f1-2)
    # Здесь можно добавить логику маппинга driver -> cuda, но проще жестко задать версию в скрипте
    echo "✅ Обнаружена CUDA. Установка версии для GPU..."
    "$ENV_PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi

echo "📥 Шаг 2/2: Синхронизация мультимодального ядра E-Corp..."
"$ENV_PIP" install \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    qwen-vl-utils>=0.0.4 \
    autoawq \
    openai-whisper \
    soundfile \
    librosa \
    pydub \
    gradio>=4.0.0 \
    python-dotenv \
    omegaconf \
    tqdm \
    noisereduce \
    pydantic \
    numpy \
    num2words \
    pymorphy3 \
    -q

echo "✅ [SUCCESS] Все библиотеки успешно установлены!"

echo "📝 Фиксация зависимостей в локальный requirements.txt..."
"$ENV_PIP" freeze --local > requirements.txt

echo "🧹 Очистка временного кэша pip..."
"$ENV_PIP" cache purge -q
echo "✅ Кэш pip очищен."

echo "⚙️ Инициализация каталогов хранения через настройки settings.py..."
"$ENV_DIR/bin/python" -c "
from config.settings import settings
from pathlib import Path
import os

paths = [
    Path(settings.HF_HOME), 
    Path(settings.VLM_DIR), 
    Path(settings.ASR_DIR), 
    Path(settings.TTS_DIR), 
    Path(settings.PROJECT_ROOT) / 'data/output_audio'
]

for p in paths:
    p.mkdir(parents=True, exist_ok=True)
print('   ✅ Все оффлайн-директории хранения успешно проинициализированы.')
"

echo "🔍 Верификация аппаратного ускорения..."
"$ENV_DIR/bin/python" -c "
import torch
if torch.cuda.is_available():
    print(f'   \033[1;32m✅ CUDA готова! Обнаружена карта: {torch.cuda.get_device_name(0)}\033[0m')
else:
    print('   \033[1;31m⚠️ CUDA недоступна! Вычисления пойдут на CPU.\033[0m')
"
