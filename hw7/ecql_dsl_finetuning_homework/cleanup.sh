#!/usr/bin/env bash
set -euo pipefail

# Переходим в директорию скрипта, чтобы пути были абсолютными
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "🧹 [CLEANUP] Очистка локального ИИ-окружения..."

# 1. Изящно тушим Ollama по актуализированному пути к PID-файлу внутри src/storage/
PID_FILE="$SCRIPT_DIR/src/storage/ollama.pid"
if [ -f "$PID_FILE" ]; then
    OLLAMA_PID=$(cat "$PID_FILE")
    if kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "⏹️ Остановка сервера Ollama (PID: $OLLAMA_PID)..."
        kill "$OLLAMA_PID" && sleep 1 || true
        kill -9 "$OLLAMA_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
else
    # На случай, если PID-файла нет, но сервер Ollama или его фоновый baken завис
    pkill -9 -f "ollama serve" 2>/dev/null || true
    pkill -9 -f "llama-server" 2>/dev/null || true
fi

# 2. Безопасно убиваем процессы обучения и генерации текущего пользователя
MY_USER=$(whoami)
for pattern in "main.py" "train.py" "seed_dataset.py" "split_dataset.py"; do
    # ИСПРАВЛЕНО: Флаг -n вместо -not -z. Исключаем PID текущего скрипта ($$)
    PIDS=$(pgrep -u "$MY_USER" -f "$pattern" | grep -v "$$" || true)
    if [ -n "$PIDS" ]; then
        echo "⏹️ Принудительное уничтожение процессов для '$pattern'..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
    fi
done

# 3. Безопасное удаление кэша компиляции Python (освобождает место, не трогая датасеты)
echo "🧹 Очистка временного кэша __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 4. Очищаем системный кэш RAM WSL (высвобождает ОЗУ ноутбука)
if [ "$EUID" -eq 0 ]; then
    sync && echo 3 > /proc/sys/vm/drop_caches
    echo "🧠 Системный кэш RAM успешно очищен."
else
    echo "ℹ️ Пропуск очистки кэша RAM (требуются права root)."
    echo "   Чтобы очистить ОЗУ, запустите: sudo $0"
fi

echo "✅ Окружение полностью очищено и готово к перезапуску!"
