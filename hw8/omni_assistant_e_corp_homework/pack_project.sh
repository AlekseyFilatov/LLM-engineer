#!/usr/bin/env bash
# Жесткий режим Bash: падать при любой ошибке внутри конвейера
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}📦 [PACK v3] Сборка, валидация и архивация Omni-Assistant проекта E-Corp...${NC}"

PROJECT_ROOT="/home/alexfil/LLM-Training"
cd "$PROJECT_ROOT"

# =====================================================================
# 1. КОНТРОЛЬ ЦЕЛОСТНОСТИ ПЕРЕД СДАЧЕЙ (Валидация логов профайлинга)
# =====================================================================
echo -e "${YELLOW}🕵️ Валидация наличия критических LLMOps артефактов...${NC}"
if [ ! -f "./data/latency_metrics.log" ]; then
    echo -e "${RED}❌ КРИТИЧЕСКИЙ СБОЙ: Лог-манифест задержек не найден по пути ./data/latency_metrics.log!${NC}"
    echo -e "💡 Сначала запустите 'python3 app.py' или './run_tests.sh' для генерации метрик."
    exit 1
fi

# =====================================================================
# 2. БЕЗОПАСНАЯ ОЧИСТКА ХОСТ-ЗАВИСИМОГО КЭША (Защита от FileNotFoundError)
# =====================================================================
if [ -d "./src/storage/silero_cache" ]; then
    echo -e "${YELLOW}🧹 Очистка хост-зависимой NoSQL базы tts_cache перед упаковкой...${NC}"
    find ./src/storage/silero_cache -type f -name "tts_cache.db*" -delete
fi

# Зачищаем временные wav-файлы ответов, оставляя саму структуру папки
if [ -d "./data/output_audio" ]; then
    find ./data/output_audio -type f -name "*.wav" -delete
fi

# =====================================================================
# 3. АВТОМАППИНГ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ДЛЯ ПРЕПОДАВАТЕЛЯ (.env.example)
# =====================================================================
echo -e "${YELLOW}📝 Обновление актуального оффлайн-шаблона .env.example...${NC}"
cat << 'EOF' > .env.example
# =====================================================================
# КОРПОРАТИВНЫЕ НАСТРОЙКИ OMNI-ASSISTANT (E-CORP SPEECH & VISION CONTEXT)
# =====================================================================
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0

OMNI_VLM_REPO=Qwen/Qwen2-VL-2B-Instruct
OMNI_ASR_REPO=openai/whisper-base
OMNI_TTS_URL=https://silero.ai

STORAGE_HF_HOME=./src/storage/hf_home_cache
STORAGE_VLM_DIR=./src/storage/hf_cache/Qwen2-VL-2B-Instruct
STORAGE_ASR_DIR=./src/storage/whisper_cache
STORAGE_TTS_DIR=./src/storage/silero_cache
STORAGE_TTS_FILE=./src/storage/silero_cache/v5_ru.pt

OMNI_MAX_NEW_TOKENS=128
OMNI_LATENCY_LOGS=./data/latency_metrics.log

OMNI_TARGET_SAMPLE_RATE=16000
OMNI_CHUNK_SECONDS=30
OMNI_OVERLAP_SECONDS=1
OMNI_MIN_CHUNK_SECONDS=0.5

VLM_MAX_RESPONSE_TOKENS=128
VLM_MAX_LATENCY_HISTORY=1000

OMNI_TTS_SPEAKER=kseniya
OMNI_TTS_SAMPLE_RATE=24000
EOF

# =====================================================================
# 4. УНИВЕРСАЛЬНАЯ ФИКСАЦИЯ ЗАВИСИМОСТЕЙ ИЗ РЕАЛЬНОГО ОКРУЖЕНИЯ
# =====================================================================
REAL_PIP_PATH="/home/alexfil/ml_env/bin/pip"

if [ -x "$REAL_PIP_PATH" ]; then
    echo -e "${GREEN}🔄 Найдено активное домашнее окружение ml_env. Фиксируем библиотеки...${NC}"
    "$REAL_PIP_PATH" freeze > requirements.txt
elif [ -x ".venv_wsl/bin/pip" ]; then
    echo -e "${GREEN}🔄 Найдено активное окружение .venv_wsl. Фиксируем библиотеки...${NC}"
    .venv_wsl/bin/pip freeze > requirements.txt
else
    echo -e "${RED}⚠️ Предупреждение: Точка рантайма pip не локализована!${NC}"
    echo "    requirements.txt не будет обновлен. Используется текущий слепок."
fi

# Имя итогового архива
ARCHIVE_NAME="omni_assistant_e_corp_homework.tar.gz"

# =====================================================================
# 5. СИЛОВАЯ АРХИВАЦИЯ С ИСКЛЮЧЕНИЕМ ТЯЖЕЛЫХ ВЕСОВ МОДЕЛЕЙ
# =====================================================================
echo -e "${YELLOW}📦 Упаковка файлов в монолитный архив tar.gz...${NC}"

tar --exclude='ml_env' \
    --exclude='.venv_wsl' \
    --exclude='storage' \
    --exclude='hf_home_cache' \
    --exclude='whisper_cache' \
    --exclude='silero_cache' \
    --exclude='hf_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.vscode' \
    --exclude='.env' \
    --exclude='*.tar.gz' \
    -czf "$ARCHIVE_NAME" \
    config \
    data \
    src \
    *.py \
    *.sh \
    requirements.txt \
    .env.example \
    README.md 2>/dev/null

# =====================================================================
# 6. ИТОГОВЫЙ КОНТРОЛЬ ВЕСА СБОРКИ
# =====================================================================
if [ -f "$ARCHIVE_NAME" ]; then
    ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
    echo -e "${GREEN}✨ [SUCCESS] Проект Omni-Assistant успешно запечен в архив: $ARCHIVE_NAME${NC}"
    echo -e "ℹ️  Финальный вес архива: ${GREEN}$ARCHIVE_SIZE${NC}. Он полностью очищен от тяжелых терабайтных весов."
    echo -e "💡 Для распаковки преподавателю достаточно выполнить: ${YELLOW}tar -xzf $ARCHIVE_NAME${NC}"
else
    echo -e "${RED}❌ Крах архивации: файл не был создан на диске.${NC}"
    exit 1
fi
