#!/bin/bash
set -euo pipefail

# Получаем абсолютный путь к директории, где лежит скрипт
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "🚨 ВНИМАНИЕ: Вы запускаете полную ликвидацию ИИ-инфраструктуры проекта!"

# Проверяем, передан ли флаг -y (force), иначе запрашиваем подтверждение
SKIP_CONFIRM=false
if [[ "${1:-}" == "-y" ]]; then
    SKIP_CONFIRM=true
fi

if [ "$SKIP_CONFIRM" = false ]; then
    # Проверяем, подключен ли терминал (чтобы не зависнуть в CI/CD)
    if [ ! -t 0 ]; then
        echo "❌ Ошибка: Терминал не обнаружен, а флаг автоматического согласия (-y) не передан."
        exit 1
    fi
    
    read -p "Вы уверены, что хотите безвозвратно удалить Ollama, веса моделей, датасеты и окружение? (y/n): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "❌ Ликвидация отменена."
        exit 0
    fi
fi

echo "🛑 Принудительная остановка фоновых серверов..."

# 1. Сначала точечно тушим Ollama по нашему PID-файлу, если он есть
PID_FILE="./storage/ollama.pid"
if [ -f "$PID_FILE" ]; then
    OLLAMA_PID=$(cat "$PID_FILE")
    if kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "⏹️ Остановка локальной Ollama (PID: $OLLAMA_PID)..."
        kill "$OLLAMA_PID" && sleep 1 || true
        kill -9 "$OLLAMA_PID" 2>/dev/null || true
    fi
fi

# 2. Остальные процессы тушим безопасно (только принадлежащие текущему пользователю)
# Исключаем имя самого скрипта из поиска, чтобы он не убил сам себя
MY_USER=$(whoami)
CURRENT_SCRIPT=$(basename "${BASH_SOURCE[0]}")

for pattern in "main.py" "train.py" "ollama serve"; do
    # Ищем PID процессов текущего пользователя, игнорируя этот скрипт
    PIDS=$(pgrep -u "$MY_USER" -f "$pattern" | grep -v "$$" || true)
    if [ -not -z "$PIDS" ]; then
        echo "⏹️ Остановка процессов для '$pattern'..."
        echo "$PIDS" | xargs kill 2>/dev/null || true
        sleep 1
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
    fi
done

# Функция безопасного удаления
safe_rm_dir() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        echo "🗑️ Удаление папки: $dir"
        rm -rf "$dir"
    else
        echo "⚠️ Папка не найдена: $dir (пропущено)"
    fi
}

safe_rm_file() {
    local file_pattern="$1"
    # Используем find вместо compgen, так как compgen может вести себя нестабильно в некоторых версиях bash
    if find . -maxdepth 2 -path "./$file_pattern" -print -quit | grep -q .; then
        echo "🗑️ Удаление файлов: $file_pattern"
        rm -f $file_pattern # Без кавычек, чтобы сработал wildcard-шаблон
    else
        echo "⚠️ Файлы не найдены: $file_pattern (пропущено)"
    fi
}

# Запускаем очистку
safe_rm_dir "storage"
safe_rm_dir "ml_env"
safe_rm_dir ".venv_wsl"

safe_rm_file "data/*.jsonl"

echo "🗑️ Удаление кэша компиляции Python (__pycache__)..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "✨ [CLEAN SUCCESS] Проект полностью очищен!"
