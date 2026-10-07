#!/usr/bin/env bash
# Жесткий режим Bash: падать при любой непредвиденной ошибке внутри конвейера
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Переходим в директорию скрипта, чтобы пути были абсолютными
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo -e "${GREEN}🧹 [CLEANUP OMNI] Очистка мультимодального ИИ-окружения...${NC}"

# =====================================================================
# 1. ИЗЯЩНАЯ ОСТАНОВКА И ОЧИСТКА ВСЕГО DOCKER COMPOSE СТЕКА
# =====================================================================
echo -e "${YELLOW}⏹️  Шаг 1/4: Изящная остановка Docker контейнеров (Triton, Qdrant, Langfuse)...${NC}"

if command -v docker-compose &> /dev/null; then
    docker-compose down --timeout 5 2>/dev/null || true
elif docker compose version &> /dev/null; then
    docker compose down --timeout 5 2>/dev/null || true
else
    echo -e "${RED}⚠️  Предупреждение: утилита docker-compose не найдена. Чистим контейнеры поштучно...${NC}"
    # Резервный силовой сценарий по маскам имен контейнеров курсового
    OMNI_CONTAINERS=$(docker ps -a -q --filter "name=omni" --filter "name=langfuse" || true)
    if [ -n "$OMNI_CONTAINERS" ]; then
        echo "$OMNI_CONTAINERS" | xargs docker rm -f 2>/dev/null || true
    fi
fi

# =====================================================================
# 2. ПРИНУДИТЕЛЬНОЕ УНИЧТОЖЕНИЕ ЗАВИСШИХ ПРОЦЕССОВ И КЛИЕНТОВ
# =====================================================================
MY_USER=$(whoami)
echo -e "${YELLOW}⏹️  Шаг 2/4: Безопасное уничтожение фоновых Python-клиентов и Gradio...${NC}"

# client_test.py, ui_client.py (Gradio), load_omni_models.py
for pattern in "client_test.py" "ui_client.py" "load_omni_models.py" "gradio"; do
    # Исключаем PID самого этого скрипта очистки ($$)
    PIDS=$(pgrep -u "$MY_USER" -f "$pattern" | grep -v "$$" || true)
    if [ -n "$PIDS" ]; then
        echo "   -> Принудительное завершение процессов для '$pattern'..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
    fi
done

# =====================================================================
# 3. БЕЗОПАСНАЯ ОЧИСТКА ВРЕМЕННОГО КЭША И МУСОРА СКАЧИВАНИЯ
# =====================================================================
echo -e "${YELLOW}🧹 Шаг 3/4: Очистка временного кэша компиляции __pycache__ и OCI-слоев...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Зачищаем недокачанные подпапки ONNX, если скрипт huggingface_hub оборвался
find src/triton_repository/ -type d -name "onnx" -exec rm -rf {} + 2>/dev/null || true

# =====================================================================
# 4. СБРОС СИСТЕМНОГО КЭША RAM WSL (Высвобождение ОЗУ хоста Windows)
# =====================================================================
echo -e "${YELLOW}🧠 Шаг 4/4: Анализ и высвобождение занятой памяти RAM ноутбука...${NC}"
# Проверяем права root или доступность drop_caches на запись
if [ "$EUID" -eq 0 ] || [ -w "/proc/sys/vm/drop_caches" ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
    echo -e "${GREEN}✅ Системный кэш RAM успешно сброшен. Память возвращена в Windows!${NC}"
else
    echo -e "${YELLOW}ℹ️  Пропуск глубокой очистки кэша RAM WSL (требуются права root).${NC}"
    echo "   Чтобы вернуть гигабайты ОЗУ хост-системе, запустите: sudo ./cleanup.sh"
fi

echo -e "${GREEN}🎉 [SUCCESS] Все порты свободны, ИИ-окружение очищено и готово к перезапуску!${NC}"
