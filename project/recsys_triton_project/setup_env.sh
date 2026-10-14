#!/usr/bin/env bash
# Жесткий режим Bash: падать при любой непредвиденной ошибке внутри конвейера
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}📦 [OMNI SETUP v2] Развертывание бесконфликтной среды Omni-поисковика...${NC}"

# =====================================================================
# 1. АВТОМАТИЧЕСКОЕ СОЗДАНИЕ ИНФРАСТРУКТУРЫ КАТАЛОГОВ
# =====================================================================
echo -e "${YELLOW}📂 Шаг 1/5: Разворачивание структуры папок проекта и Triton...${NC}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Создаем структуру Model Repository Triton Server
TRITON_REPO="./src/triton_repository"
mkdir -p "$TRITON_REPO/omni_search_orchestrator/1"
mkdir -p "$TRITON_REPO/clip_embedder/1"
mkdir -p "$TRITON_REPO/qwen_vlm_rag/1"

# Создаем папки персистентного хранения данных баз и картинок
mkdir -p "./src/storage/qdrant_data"
mkdir -p "./src/storage/postgres_data"
mkdir -p "./data/catalog_images"

echo "✅ Архитектурный каркас папок успешно зафиксирован."

# =====================================================================
# 2. АВТОЛОКАЛИЗАЦИЯ ИЛИ ИНИЦИАЛИЗАЦИЯ VENV ОКРУЖЕНИЯ
# =====================================================================
echo -e "${YELLOW}🕵️ Шаг 2/5: Локализация виртуального окружения Python...${NC}"
ENV_DIR=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo -e "✅ Зафиксировано активное окружение в терминале: ${GREEN}$(basename "$VIRTUAL_ENV")${NC}"
    ENV_DIR="$VIRTUAL_ENV"
elif [ -d ".venv" ]; then
    echo "🔄 Найдена папка .venv. Используем её..."
    ENV_DIR="$PROJECT_ROOT/.venv"
elif [ -d "ml_env" ]; then
    echo "🔄 Найдена папка ml_env. Используем её..."
    ENV_DIR="$PROJECT_ROOT/ml_env"
else
    echo "🔄 Локальное окружение не найдено. Создаем чистый .venv..."
    if ! python3 -m venv .venv 2>/dev/null; then
        echo -e "${RED}❌ Ошибка: В системе отсутствует пакет python3-venv.${NC}"
        echo "💡 Выполните на хосте: sudo apt update && sudo apt install -y python3-venv"
        exit 1
    fi
    ENV_DIR="$PROJECT_ROOT/.venv"
fi

ENV_PIP="$ENV_DIR/bin/pip"
ENV_PYTHON="$ENV_DIR/bin/python"

echo "🔄 Обновление менеджера пакетов pip..."
"$ENV_PIP" install --upgrade pip -q

# =====================================================================
# 3. КОНТРОЛЬ И СИНХРОНИЗАЦИЯ ВЫЧИСЛИТЕЛЬНОГО БЭКЕНДА (PYTORCH)
# =====================================================================
echo -e "${YELLOW}📥 Шаг 3/5: Контроль наличия вычислительного ядра PyTorch...${NC}"

if "$ENV_PYTHON" -c "import torch" &>/dev/null; then
    echo -e "${GREEN}✅ PyTorch уже предустановлен в этой среде. Пропускаем загрузку.${NC}"
else
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠️ nvidia-smi не найден. Установка PyTorch (CPU-версия)...${NC}"
        "$ENV_PIP" install torch torchvision \
            --extra-index-url https://download.pytorch.org/whl/cpu -q
    else
        # Определяем версию CUDA из nvidia-smi
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo -e "${GREEN}✅ GPU обнаружен (драйвер: ${CUDA_VERSION}). Установка PyTorch с CUDA 12.6...${NC}"
        "$ENV_PIP" install torch torchvision \
            --extra-index-url https://download.pytorch.org/whl/cu126 -q || {
                echo -e "${YELLOW}⚠️ PyTorch-индекс недоступен. Переключаемся на стандартный PyPI...${NC}"
                "$ENV_PIP" install torch torchvision -q
            }
    fi
fi

# =====================================================================
# 4. РАЗРЕШЕНИЕ КОНФЛИКТА: СНОС И БЕЗОПАСНАЯ СИНХРОНИЗАЦИЯ БИБЛИОТЕК
# =====================================================================
echo -e "${YELLOW}🧹 Шаг 4/5: Принудительное удаление конфликтующих метапакетов...${NC}"
# Полностью вырезаем сбойный sentence-transformers, ломающий граф transformers 4.x
"$ENV_PIP" uninstall -y sentence-transformers huggingface-hub transformers -q || true

echo -e "${YELLOW}📥 Шаг 5/5: Чистая установка Omni-стека (Triton, Qdrant, Gradio, Langfuse)...${NC}"
# Устанавливаем жестко зафиксированные, обратно совместимые версии
"$ENV_PIP" install \
    "tritonclient[all]>=2.54.0" \
    "numpy>=1.24.0,<2.0.0" \
    "transformers>=4.36.0,<4.58.0" \
    "huggingface-hub>=0.24.0,<1.0.0" \
    "gradio>=4.0.0" \
    "qdrant-client>=1.9.0" \
    "langfuse>=2.0.0" \
    "gradio>=4.0.0" \
    python-dotenv \
    pydantic-settings \
    langgraph \
    pymorphy3 \
    spacy \
    scikit-learn \
    tqdm \
    Pillow \
    scipy \
    requests \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org -q

echo -e "${GREEN}✅ [SUCCESS] Все библиотеки Omni-стека успешно синхронизированы без конфликтов!${NC}"

echo "📝 Фиксация стабильного слепка зависимостей в requirements.txt..."
"$ENV_PIP" freeze --local > requirements.txt

echo "🧹 Очистка временного кэша pip..."
"$ENV_PIP" cache purge -q
echo -e "${GREEN}✅ Кэш pip успешно очищен. Локальное рабочее пространство полностью готово!${NC}"
