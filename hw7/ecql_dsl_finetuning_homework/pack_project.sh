#!/usr/bin/env bash
# Жесткий режим Bash: падать при любой ошибке внутри конвейера
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}📦 [PACK v2] Сборка, валидация и архивация Fine-tuning проекта E-Corp...${NC}"

PROJECT_ROOT="/home/alexfil/LLM-Training"
cd "$PROJECT_ROOT"

# =====================================================================
# 1. КОНТРОЛЬ ЦЕЛОСТНОСТИ (Защита от отправки пустого проекта)
# =====================================================================
echo -e "${YELLOW}🕵️ Валидация наличия критических артефактов...${NC}"
if [ ! -f "./data/ecql_test.jsonl" ]; then
    echo -e "${RED}❌ КРИТИЧЕСКИЙ СБОЙ: Отложенный тестовый датасет не найден по пути ./data/ecql_test.jsonl!${NC}"
    echo -e "💡 Сначала запустите верификацию, чтобы сгенерировать тест-кейсы."
    exit 1
fi

# =====================================================================
# 2. АВТОМАППИНГ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ (.env.example)
# =====================================================================
echo -e "${YELLOW}📝 Обновление оффлайн-шаблона .env.example...${NC}"
cat << 'EOF' > .env.example
# =====================================================================
# СИСТЕМНЫЕ НАСТРОЙКИ (DSL GENERATOR & FINE-TUNING)
# =====================================================================

# Пути к локальным директориям проекта на диске (ext4/WSL)
MODEL_NAME="./src/storage/hf_cache/Qwen2.5-Coder-7B-Instruct"
HF_HOME="./src/storage/models_cache"

# Параметры батчинга и токенизации для инференса на GPU (bfloat16)
EVAL_BATCH_SIZE=1
MAX_NEW_TOKENS=128

# Лимиты для итеративного генератора датасетов ECQL
TOTAL_TARGET_SAMPLES=200
SIMILARITY_THRESHOLD=0.75
MAX_ITERATIONS=50
EOF

# =====================================================================
# 3. УНИВЕРСАЛЬНАЯ ФИКСАЦИЯ ЗАВИСИМОСТЕЙ (Без капризов команды source)
# =====================================================================
if [ -x ".venv_wsl/bin/pip" ]; then
    echo -e "${GREEN}🔄 Найдено активное окружение .venv_wsl. Фиксируем библиотеки...${NC}"
    .venv_wsl/bin/pip freeze > requirements.txt
elif [ -x "ml_env/bin/pip" ]; then
    echo -e "${GREEN}🔄 Найдено активное окружение ml_env. Фиксируем библиотеки...${NC}"
    ml_env/bin/pip freeze > requirements.txt
else
    echo -e "${RED}⚠️ Предупреждение: Ни ml_env, ни .venv_wsl не найдены в корне проекта!${NC}"
    echo "    requirements.txt не будет обновлен. Используется старый слепок."
fi

# Новое имя архива
ARCHIVE_NAME="ecql_dsl_finetuning_homework.tar.gz"

# =====================================================================
# 4. СИЛОВАЯ АРХИВАЦИЯ С УЧЕТОМ ОФФЛАЙН-ИНСТРУМЕНТАРИЯ И СХЕМЫ
# =====================================================================
echo -e "${YELLOW}📦 Упаковка файлов в монолитный архив tar.gz...${NC}"

# Явно перечисляем все файлы, которые мы создали и починили
# ДОБАВЛЕНЫ: папка config (где лежит schema.yaml), все *.py файлы, *.sh и файлы данных data/
tar --exclude='ml_env' \
    --exclude='.venv_wsl' \
    --exclude='storage' \
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
# 5. ИТОГОВЫЙ КОНТРОЛЬ СБОРКИ
# =====================================================================
if [ -f "$ARCHIVE_NAME" ]; then
    ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
    echo -e "${GREEN}✨ [SUCCESS] Проект E-Corp успешно запечен в архив: $ARCHIVE_NAME${NC}"
    echo -e "ℹ️  Финальный вес архива: ${GREEN}$ARCHIVE_SIZE${NC}. Он полностью очищен от тяжелых терабайтных весов."
    echo -e "💡 Для распаковки достаточно выполнить: ${YELLOW}tar -xzf $ARCHIVE_NAME${NC}"
else
    echo -e "${RED}❌ Крах архивации: файл не был создан на диске.${NC}"
    exit 1
fi
