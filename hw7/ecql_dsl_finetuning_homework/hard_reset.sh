#!/usr/bin/env bash
set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}🚨 ВНИМАНИЕ: Вы запускаете полную очистку проекта Fine-tuning!${NC}"
echo "Будут удалены:"
echo "  - Локальная папка storage/"
echo "  - Сгенерированные датасеты JSONL в data/"
echo "  - Виртуальные окружения (.venv_wsl/, ml_env/)"
echo "  - Кэш моделей Hugging Face (все найденные копии)"
echo "  - Кэши Python (__pycache__)"
echo

# Обработка автоматического согласия через флаг -y
SKIP_CONFIRM=false
if [[ "${1:-}" == "-y" ]]; then
    SKIP_CONFIRM=true
fi

if [ "$SKIP_CONFIRM" = false ]; then
    # Проверяем наличие терминала, чтобы не зависнуть в автоматических пайплайнах
    if [ ! -t 0 ]; then
        echo -e "${RED}❌ Ошибка: Терминал не обнаружен. Для автоматической очистки запустите: $0 -y${NC}"
        exit 1
    fi
    
    read -p "Вы уверены, что хотите продолжить? (y/Y): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo -e "${YELLOW}❌ Очистка отменена пользователем.${NC}"
        exit 0
    fi
fi

echo
echo -e "${YELLOW}🗑️ Начинаем процесс очистки...${NC}"

# --- Локальные папки проекта ---

if [ -d "storage" ]; then
    echo "🗑️ Удаление локальной папки хранения и логов..."
    rm -rf storage/
else
    echo "ℹ️ Папка storage/ не найдена — пропускаем."
fi

if [ -d "data" ]; then
    echo "🗑️ Удаление сгенерированных датасетов JSONL (оставляем папку data/ пустой)..."
    # Безопасная проверка: '|| true' защищает от жесткого падения set -o pipefail
    if find data/ -maxdepth 1 -type f -name "*.jsonl" -print -quit | grep -q . 2>/dev/null || true; then
        rm -f data/*.jsonl
        echo "✅ Файлы .jsonl успешно удалены."
    else
        echo "ℹ️ В папке data/ нет файлов .jsonl — пропускаем удаление."
    fi
else
    echo "ℹ️ Папка data/ не найдена — пропускаем."
fi

# Удаление виртуальных окружений
for env_dir in ".venv_wsl" "ml_env"; do
    if [ -d "$env_dir" ]; then
        echo "🗑️ Удаление виртуального окружения: $env_dir/"
        rm -rf "$env_dir"
    else
        echo "ℹ️ Папка $env_dir/ не найдена — пропускаем."
    fi
done

# --- Кэш Hugging Face ---

huggingface_paths=(
    "/mnt/d/ai_models/huggingface"
    "$HOME/.cache/huggingface"
)

hf_removed=false
for hf_path in "${huggingface_paths[@]}"; do
    if [ -d "$hf_path" ]; then
        echo "🗑️ Обнаружен кэш Hugging Face: $hf_path"
        echo "    Удаление скачанных весов LLM моделей..."
        rm -rf "$hf_path"
        hf_removed=true
        # break удален: зачищаем все места, где могут лежать тяжелые гигабайты весов
    fi
done

if [ "$hf_removed" = false ]; then
    echo "ℹ️ Кэш Hugging Face не найден ни в одном из ожидаемых путей — пропускаем."
fi

# --- Очистка __pycache__ ---

echo "🗑️ Очистка кэша компиляции Python (__pycache__)..."
# -prune оптимизирует поиск, удаляя папки целиком без рекурсивного обхода внутренностей
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name "*.egg-info" \) -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo -e "${GREEN}✨ Проект успешно сброшен до исходного состояния!${NC}"
echo "Окружение, веса и базы данных удалены."
echo
echo "ℹ️ Теперь вы можете запустить: ./setupenv.sh для чистой повторной инициализации."

