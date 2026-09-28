#!/usr/bin/env bash
# Жесткий режим Bash: падать при любой непредвиденной ошибке внутри конвейера
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}📦 [PACK v4] Сборка, валидация и архивация Triton Inference конвейера...${NC}"

PROJECT_ROOT="/home/alexfil/LLM-Training"
cd "$PROJECT_ROOT"

# =====================================================================
# 1. КОНТРОЛЬ ЦЕЛОСТНОСТИ (Проверка верификации клиента)
# =====================================================================
echo -e "${YELLOW}🕵️ Валидация наличия критических LLMOps артефактов...${NC}"
if [ ! -f "ui_client.py" ] && [ ! -f "verify_client.py" ]; then
    echo -e "${RED}❌ КРИТИЧЕСКИЙ СБОЙ: Верификационный скрипт клиента не найден в корне проекта!${NC}"
    echo -e "💡 Убедитесь, что файлы ui_client.py или verify_client.py находятся в $PROJECT_ROOT"
    exit 1
fi

# =====================================================================
# 2. БЕЗОПАСНАЯ ОЧИСТКА ВРЕМЕННОГО КЭША (Защита от FileNotFoundError)
# =====================================================================
echo -e "${YELLOW}🧹 Очистка временных логов и питоновских кэшей...${NC}"
# Зачищаем локальный кэш питона во избежание мусора в архиве
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =====================================================================
# 3. АВТОМАППИНГ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ (.env.example)
# =====================================================================
echo -e "${YELLOW}📝 Создание актуального оффлайн-шаблона конфигурации .env.example...${NC}"
cat << 'EOF' > .env.example
# =====================================================================
# КОНФИГУРАЦИЯ СРЕДЫ TRITON INFERENCE SERVER (OFFLINE PRODUCTION)
# =====================================================================
# Параметры NVIDIA Triton Server
TRITON_GRPC_URL=localhost:8001
TRITON_MODEL_NAME=clip_embedder
TRITON_VLM_NAME=paligemma_vlm

# Параметры Векторной Базы Данных Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6334
QDRANT_COLLECTION=catalog

# Параметры Микросервиса Распознавания Речи Whisper
WHISPER_URL=http://localhost:8003/v1/audio/transcriptions

# Параметры Системы Мониторинга Langfuse LLMOps
LANGFUSE_HOST=http://localhost:3000

# Локальные пути внутри репозитория проекта
LOCAL_TOKENIZER_PATH=~/LLM-Training/src/clip_tokenizer

LOCAL_PALIGEMMA_TOKENIZER_PATH=~/LLM-Training/src/paligemma_tokenizer

# LOCAL_RERANKER_TOKENIZER_PATH=~/LLM-Training/src/reranker_tokenizer
# Локальные пути к новым русскоязычным токенизаторам хоста
LOCAL_TOKENIZER_PATH="/home/alexfil/LLM-Training/src/paraphrase_tokenizer"
LOCAL_RERANKER_TOKENIZER_PATH="/home/alexfil/LLM-Training/src/reranker_tokenizer"

# Конфигурация Веб-интерфейса Gradio
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
#GRADIO_SHARE=False

LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="http://localhost:3000"

TRITON_GRPC_URL="127.0.0.1:8001"
EOF

# =====================================================================
# 4. УНИВЕРСАЛЬНАЯ ФИКСАЦИЯ ЗАВИСИМОСТЕЙ ИЗ РЕАЛЬНОГО ОКРУЖЕНИЯ
# =====================================================================
REAL_PIP_PATH="/home/alexfil/LLM-Training/ml_env/bin/pip"

if [ -x "$REAL_PIP_PATH" ]; then
    echo -e "${GREEN}🔄 Найдено активное домашнее окружение ml_env. Фиксируем библиотеки...${NC}"
    "$REAL_PIP_PATH" freeze > requirements.txt
elif [ -x "ml_env/bin/pip" ]; then
    echo -e "${GREEN}🔄 Найдено локальное окружение ml_env. Фиксируем библиотеки...${NC}"
    ml_env/bin/pip freeze > requirements.txt
else
    echo -e "${RED}⚠️ Предупреждение: Точка рантайма pip не локализована!${NC}"
    echo "    requirements.txt не будет обновлен. Используется текущий слепок."
fi

# Имя итогового архива лабораторной работы
ARCHIVE_NAME="recsys_triton_project.tar.gz"

# =====================================================================
# 5. СИЛОВАЯ АРХИВАЦИЯ С ИСКЛЮЧЕНИЕМ ТЯЖЕЛЫХ ВЕСОВ МОДЕЛЕЙ (ONNX/LLM)
# =====================================================================
echo -e "${YELLOW}📦 Упаковка файлов в монолитный архив tar.gz...${NC}"

# Жестко исключаем тяжелое окружение ml_env и бинарные веса .onnx
# Преподаватель получит структуру папок, конфиги pbtxt и код оркестратора
tar --exclude='ml_env' \
    --exclude='paraphrase_model_pytorch' \
    --exclude='paraphrase_tokenizer' \
    --exclude='reranker_tokenizer' \
    --exclude='clip_tokenizer' \
    --exclude='.venv' \
    --exclude='*.onnx' \
    --exclude='*.onnx_data' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.vscode' \
    --exclude='.env' \
    --exclude='*.tar.gz' \
    -czf "$ARCHIVE_NAME" \
    config \
    data \
    tests \
    src \
    *.py \
    *.sh \
    requirements.txt \
    .env.example \
    README.md 2>/dev/null || true

# =====================================================================
# 6. ИТОГОВЫЙ КОНТРОЛЬ ВЕСА СБОРКИ
# =====================================================================
if [ -f "$ARCHIVE_NAME" ]; then
    ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
    echo -e "${GREEN}✨ [SUCCESS] Проект Triton BLS успешно запечен в архив: $ARCHIVE_NAME${NC}"
    echo -e "ℹ️  Финальный вес архива: ${GREEN}$ARCHIVE_SIZE${NC}. Он полностью очищен от тяжелых терабайтных весов моделей."
    echo -e "💡 Для распаковки преподавателю достаточно выполнить: ${YELLOW}tar -xzf $ARCHIVE_NAME${NC}"
else
    echo -e "${RED}❌ Крах архивации: файл не был создан на диске.${NC}"
    exit 1
fi
