#!/usr/bin/env bash
# Жесткий режим Bash: падать при любых ошибках
set -euo pipefail

# Цвета для вывода терминала
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}🚨 ВНИМАНИЕ: Вы запускаете ПОЛНУЮ ОЧИСТКУ мультимодального проекта E-Corp!${NC}"
echo "Будут безвозвратно удалены:"
echo "  - Локальный оффлайн-кэш ИИ-моделей в src/storage/ (Qwen2-VL, Whisper, Silero v5)"
echo "  - Локальная NoSQL база данных кэширования звука tts_cache.db"
echo "  - Метрики LLMOps профайлера (data/latency_metrics.log) и сгенерированные .wav файлы"
echo "  - Изолированное виртуальное окружение Python (ml_env)"
echo "  - Скрытые кэши Python (__pycache__, .pytest_cache)"
echo

# Обработка флага автоматического согласия -y
SKIP_CONFIRM=false
if [[ "${1:-}" == "-y" ]]; then
    SKIP_CONFIRM=true
fi

if [ "$SKIP_CONFIRM" = false ]; then
    # Проверяем наличие терминала, чтобы не зависнуть в фоне
    if [ ! -t 0 ]; then
        echo -e "${RED}❌ Ошибка: Терминал не обнаружен. Для автоматического запуска используйте: $0 -y${NC}"
        exit 1
    fi
    
    read -p "Вы действительно хотите сбросить проект до исходного состояния? (y/Y): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo -e "${YELLOW}❌ Операция очистки отменена пользователем.${NC}"
        exit 0
    fi
fi

echo
echo -e "${YELLOW}🗑️ Начинаем глубокую очистку локального периметра...${NC}"

# --- 1. Очистка пользовательских данных и логов профайлинга ---
if [ -d "data" ]; then
    echo "🗑️ Зачистка папки data/ (вырезаем логи и генерируемые аудио-файлы)..."
    rm -f data/latency_metrics.log 2>/dev/null || true
    if [ -d "data/output_audio" ]; then
        find data/output_audio -type f -name "*.wav" -delete 2>/dev/null || true
        echo "    ✅ Временные файлы .wav успешно удалены."
    fi
fi

# --- 2. Очистка оффлайн-хранилища тяжелых весов моделей E-Corp ---
if [ -d "src/storage" ]; then
    echo "🗑️ Удаление локального оффлайн-кэша моделей и NoSQL баз в src/storage/..."
    # Удаляем внутреннее содержимое, сохраняя структуру каталога src/
    rm -rf src/storage/* 2>/dev/null || true
    echo "    ✅ Папка src/storage/ полностью очищена от гигабайтных весов."
else
    echo "ℹ️ Папка src/storage/ не найдена — пропускаем."
fi

# --- 3. Удаление изолированного виртуального окружения ml_env ---
# Проверяем как локальную папку, так и Ваш реальный абсолютный домашний путь
hunted_envs=(
    "ml_env"
    ".venv_wsl"
    "/home/alexfil/ml_env"
)

for env_path in "${hunted_envs[@]}"; do
    if [ -d "$env_path" ]; then
        echo "🗑️ Удаление виртуального окружения по пути: $env_path/"
        rm -rf "$env_path"
        echo "    ✅ Окружение успешно уничтожено."
    fi
done

# --- 4. Очистка кэша компиляции байт-кода Python ---
echo "🗑️ Очистка скрытых системных кэшей Python (__pycache__)..."
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name "*.egg-info" \) -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo -e "${GREEN}✨ [CLEANUP COMPLETE] Проект успешно сброшен до исходного состояния!${NC}"
echo "Все тяжелые оффлайн-модели, логи задержек и окружения стерты."
echo
echo -e "💡 Для повторного чистого развертывания запустите: ${YELLOW}./setup_env.sh${NC}"
