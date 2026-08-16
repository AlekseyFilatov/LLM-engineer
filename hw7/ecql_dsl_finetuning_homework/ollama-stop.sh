#!/bin/bash
set -euo pipefail

echo "⏹️ [OLLAMA STOP] Остановка локального ИИ-сервера компании E-Corp..."

# Переходим в директорию скрипта
cd "$(dirname "$0")" || { echo "❌ Не удалось перейти в директорию скрипта"; exit 1; }

BASE_DIR=$(pwd)
PID_FILE="$BASE_DIR/storage/ollama.pid"

# 1. Проверяем, существует ли файл с PID
if [ ! -f "$PID_FILE" ]; then
    echo "⚠️ PID-файл не найден ($PID_FILE). Возможно, сервер уже остановлен."
    exit 0
fi

OLLAMA_PID=$(cat "$PID_FILE")

# 2. Проверяем, запущен ли процесс с таким PID
if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "⚠️ Процесс с PID $OLLAMA_PID не существует. Удаляем устаревший PID-файл."
    rm -f "$PID_FILE"
    exit 0
fi

# 3. Пытаемся изящно остановить процесс (SIGTERM)
echo "⏳ Отправка сигнала остановки процессу $OLLAMA_PID..."
kill "$OLLAMA_PID"

# 4. Ожидаем завершения процесса (максимум 10 секунд)
STOPPED=false
for i in {1..10}; do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        STOPPED=true
        break
    fi
    sleep 1
done

# 5. Если процесс завис — добиваем его жестко (SIGKILL)
if [ "$STOPPED" = false ]; then
    echo "⚠️ Процесс не завершился вовремя. Принудительное уничтожение (SIGKILL)..."
    kill -9 "$OLLAMA_PID" 2>/dev/null || true
fi

# 6. Очищаем PID-файл
rm -f "$PID_FILE"
echo "✅ Сервер Ollama успешно остановлен."

# chmod +x ollama-stop.sh