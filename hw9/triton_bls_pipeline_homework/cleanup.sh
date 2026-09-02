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

echo -e "${GREEN}🧹 [CLEANUP v4] Очистка локального ИИ-окружения Triton & Docker...${NC}"

# =====================================================================
# 1. ИЗЯЩНАЯ ОСТАНОВКА КОНТЕЙНЕРОВ TRITON И PERF ANALYZER
# =====================================================================
echo -e "${YELLOW}⏹️ Поиск и остановка контейнеров Triton Server и SDK...${NC}"

# Ищем запущенные контейнеры по имени образов nvcr.io/nvidia/tritonserver
TRITON_CONTAINERS=$(docker ps -q --filter "ancestor=nvcr.io/nvidia/tritonserver:25.12-py3" \
                              --filter "ancestor=nvcr.io/nvidia/tritonserver:25.12-py3-sdk" || true)

if [ -n "$TRITON_CONTAINERS" ]; then
    echo -e "🐳 Найдено активных контейнеров: $(echo "$TRITON_CONTAINERS" | wc -l). Тушим..."
    # Удаляем принудительно, высвобождая порты 8000, 8001, 8002
    echo "$TRITON_CONTAINERS" | xargs docker rm -f 2>/dev/null || true
else
    echo "ℹ️ Активных ИИ-контейнеров Triton не обнаружено."
fi

# =====================================================================
# 2. ПРИНУДИТЕЛЬНОЕ УНИЧТОЖЕНИЕ ЗАВИСШИХ ПРОЦЕССОВ ТЕСТИРОВАНИЯ
# =====================================================================
MY_USER=$(whoami)
echo -e "${YELLOW}⏹️ Безопасная остановка фоновых Python-клиентов и тестов...${NC}"

for pattern in "client_test.py" "verify_client.py" "load_onnx.py" "load_qwen_stable.py"; do
    # Исключаем PID самого этого скрипта очистки ($$)
    PIDS=$(pgrep -u "$MY_USER" -f "$pattern" | grep -v "$$" || true)
    if [ -n "$PIDS" ]; then
        echo "   -> Принудительное уничтожение процессов для '$pattern'..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
    fi
done

# =====================================================================
# 3. БЕЗОПАСНАЯ ОЧИСТКА ВРЕМЕННОГО КЭША И МУСОРА КОМПИЛЯЦИИ
# =====================================================================
echo -e "${YELLOW}🧹 Очистка временного кэша компиляции __pycache__...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Чистим временные OCI/Skopeo слои, если они застряли при скачивании
if [ -d "./src/triton_repository/text_generator/1/onnx" ]; then
    echo "🧹 Удаление недокачанных подпапок ONNX в генераторе..."
    rm -rf "./src/triton_repository/text_generator/1/onnx"
fi

# =====================================================================
# 4. СБРОС КЭША ОПЕРАТИВНОЙ ПАМЯТИ WSL (Очищает ОЗУ ноутбука)
# =====================================================================
echo -e "${YELLOW}🧠 Анализ и высвобождение занятой памяти RAM...${NC}"
if [ "$EUID" -eq 0 ] || [ "${1:-}" = "--sudo" ] || [ -w "/proc/sys/vm/drop_caches" ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
    echo -e "${GREEN}✅ Системный кэш RAM успешно сброшен. Ноутбук дышит свободно!${NC}"
else
    echo "ℹ️ Пропуск глубокой очистки кэша RAM WSL (требуются права root)."
    echo "   Чтобы вернуть терабайты ОЗУ хост-системе Windows, запустите: sudo ./cleanup.sh"
fi

echo -e "${GREEN}🎉 [SUCCESS] Окружение Triton конвейера полностью очищено и готово к перезапуску!${NC}"
