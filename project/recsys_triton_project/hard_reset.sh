#!/usr/bin/env bash
# =====================================================================
# КОРПОРАТИВНЫЙ СКРИПТ ТОТАЛЬНОЙ ОЧИСТКИ СИСТЕМЫ (HARD RESET)
# Полная ликвидация ИИ-инфраструктуры Omni-поисковика и Docker-слоев.
# =====================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Переходим в директорию скрипта, чтобы пути были абсолютными
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo -e "${RED}🚨 [WARNING] ВНИМАНИЕ: Вы запускаете тотальный ХАРД-РЕЗЕТ проекта Omni-Search!${NC}"
echo "Это приведет к полной очистке диска ноутбука от тяжелых Docker-образов Triton Server,"
echo "векторной базы Qdrant, логов Langfuse, а также удалит все скачанные веса ONNX-моделей."
read -p "Вы абсолютно уверены, что хотите уничтожить окружение? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo -e "${GREEN}❌ Ликвидация успешно отменена пользователем.${NC}"
    exit 0
fi

echo -e "${YELLOW}🛑 [1/5] Принудительная остановка всех фоновых ИИ-процессов и Docker-сетей...${NC}"
# Гасим локальные процессы инференса и тестов, если они зависли
pkill -9 -f tritonserver 2>/dev/null || true
pkill -9 -f perf_analyzer 2>/dev/null || true
pkill -9 -f "ui_client.py" 2>/dev/null || true
pkill -9 -f "gradio" 2>/dev/null || true

# Останавливаем стек через Docker Compose с удалением связанных томов (volumes)
if command -v docker-compose &> /dev/null; then
    docker-compose down -v --timeout 2 2>/dev/null || true
elif docker compose version &> /dev/null; then
    docker compose down -v --timeout 2 2>/dev/null || true
fi

# Принудительно находим и удаляем любые контейнеры нашего курсового проекта
OMNI_CONTAINERS=$(docker ps -a -q --filter "name=triton_server_omni" \
                                --filter "name=qdrant_db_omni" \
                                --filter "name=langfuse" || true)
if [ -n "$OMNI_CONTAINERS" ]; then
    echo "   Удаление остаточных зависших ИИ-контейнеров..."
    echo "$OMNI_CONTAINERS" | xargs docker rm -f 2>/dev/null || true
fi

echo -e "${YELLOW}🗑️  [2/5] Деструктивное удаление тяжелых Docker-образов (Освобождение ~30-50 ГБ)...${NC}"
# Удаляем исходный 19-гигабайтный образ Triton и SDK
docker rmi -f nvcr.io/nvidia/tritonserver:25.12-py3 2>/dev/null || true
docker rmi -f nvcr.io/nvidia/tritonserver:25.12-py3-sdk 2>/dev/null || true
# Удаляем смежные служебные образы Qdrant, Postgres и Langfuse
docker rmi -f qdrant/qdrant:latest 2>/dev/null || true
docker rmi -f langfuse/langfuse:latest 2>/dev/null || true
docker rmi -f postgres:16-alpine 2>/dev/null || true

# Очищаем глубокий скрытый кэш сборщика Docker Buildx (зависшие apt-get слои)
docker builder prune -a -f

echo -e "${YELLOW}🧹 [3/5] Тотальная зачистка неиспользуемых системных ресурсов Docker...${NC}"
# Силовой System Prune удаляет все висящие виртуальные сети, бесхозные тома и слои
docker system prune -a -f --volumes

echo -e "${YELLOW}📂 [4/5] Деструктивная очистка локальных хранилищ, весов моделей и venv...${NC}"
# Полное удаление виртуального окружения Python
if [ -d "ml_env" ]; then rm -rf ml_env; fi
if [ -d ".venv" ]; then rm -rf .venv; fi

# Тотальное уничтожение баз данных и весов моделей. Папки будут воссозданы пустыми.
rm -rf ./src/storage/qdrant_data/* 2>/dev/null || true
rm -rf ./src/storage/postgres_data/* 2>/dev/null || true

# Удаляем скачанные тяжелые бинарники ONNX-моделей (CLIP и Whisper)
find ./src/triton_repository/ -type f -name "*.onnx" -delete 2>/dev/null || true
find ./src/triton_repository/ -type f -name "*.onnx_data" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo -e "${YELLOW}🔒 [5/5] Принудительный сброс кэша оперативной памяти ядра WSL...${NC}"
# Полный сброс drop_caches ядра Linux для возврата RAM в Windows
if [ "$EUID" -eq 0 ] || [ -w "/proc/sys/vm/drop_caches" ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
fi

echo -e "${GREEN}="*70"${NC}"
echo -e "${GREEN}🏁 [SUCCESS] ТОТАЛЬНЫЙ ХАРД-РЕЗЕТ И КЛИНИНГ СИСТЕМЫ УСПЕШНО ЗАВЕРШЕНЫ!${NC}"
echo "Все тяжелые Docker-образы, базы Qdrant/Langfuse и веса ONNX стерты."
echo "Диск и оперативная память вашего ноутбука полностью свободны для новой сборки."
echo -e "${GREEN}="*70"${NC}"