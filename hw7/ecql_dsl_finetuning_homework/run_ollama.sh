#!/bin/bash
set -euo pipefail

echo "🚀 [OLLAMA START] Запуск локального ИИ-сервера компании E-Corp..."

# Переходим в директорию скрипта
cd "$(dirname "$0")" || { echo "❌ Не удалось перейти в директорию скрипта"; exit 1; }

# Делаем пути абсолютными, чтобы Ollama гарантированно не запуталась
BASE_DIR=$(pwd)
mkdir -p "$BASE_DIR/storage/ollama_models"


export OLLAMA_MODELS="$BASE_DIR/storage/ollama_models"
PID_FILE="$BASE_DIR/storage/ollama.pid"
LOG_FILE="$BASE_DIR/storage/ollama.log"
BIN_FILE="$BASE_DIR/storage/ollama_bin/ollama"

# 1. Проверяем наличие бинарного файла
if [ ! -f "$BIN_FILE" ]; then
    echo "❌ Бинарный файл Ollama не найден: $BIN_FILE"
    exit 1
fi

# 2. Защита от повторного запуска
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️ Ollama уже запущена с PID: $OLD_PID. Повторный запуск отменен."
        exit 0
    fi
fi

# 3. Проверяем, не занят ли порт кем-то еще
if curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
    echo "❌ Ошибка: Порт 11434 уже занят другим процессом или системной Ollama."
    exit 1
fi

# Обязательно указываем путь к нашей локальной папке библиотек,
# чтобы llama-server мог подгрузить модули поддержки вашей видеокарты
# export LD_LIBRARY_PATH="$BASE_DIR/storage/ollama_bin/lib:${LD_LIBRARY_PATH:-}"
BASE_DIR=$(pwd)
# Указываем ОС проверять обе локальные папки на наличие .so файлов (бэкендов вычислений)
export LD_LIBRARY_PATH="$BASE_DIR/storage/ollama_bin:$BASE_DIR/storage/ollama_bin/lib:${LD_LIBRARY_PATH:-}"

# Устанавливаем переменную моделей (ваш рабочий код)
export OLLAMA_MODELS="$BASE_DIR/storage/ollama_models"

BASE_DIR=$(pwd)

# 4. Запускаем Ollama в фоне
nohup "$BIN_FILE" serve > "$LOG_FILE" 2>&1 &
OLLAMA_PID=$!

# Записываем PID сразу
echo "$OLLAMA_PID" > "$PID_FILE"

# 5. Динамическое ожидание готовности (до 15 секунд)
echo "⏳ Ожидание инициализации API сервера..."
API_READY=false
for i in {1..15}; do
    if curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
        API_READY=true
        break
    fi
    # Если процесс внезапно умер во время старта — падаем сразу
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "❌ Процесс Ollama аварийно завершился. Проверьте логи: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 1
done

if [ "$API_READY" = true ]; then
    echo "✅ Ollama успешно запущена и готова к работе!"
    echo "📂 PID: $OLLAMA_PID"
    echo "📂 Логи доступны в: $LOG_FILE"
else
    echo "❌ Ошибка: Сервер запустился, но API не ответило за 15 секунд."
    kill "$OLLAMA_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
fi

# chmod +x run_ollama.sh