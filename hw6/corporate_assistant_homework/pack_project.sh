#!/bin/bash
echo "📦 Сборка и архивация проекта для сдачи..."

# 1. Автоматическое создание шаблона переменных окружения для преподавателя
echo "📝 Создание файла .env.example..."
cat << EOF > .env.example
# Токен Hugging Face (оставьте пустым для публичных моделей вроде Qwen2.5)
HF_TOKEN=

# Путь к кэшу моделей (внутри WSL рекомендуется использовать домашнюю директорию)
HF_HOME=./storage/models_cache

# Путь к локальной векторной базе данных ChromaDB
CHROMA_DB_DIR=./storage/chroma_db
EOF

# 2. Универсальная фиксация зависимостей (ищет .venv_wsl или ml_env)
if [ -d ".venv_wsl" ]; then
    echo "🔄 Найдено окружение .venv_wsl. Фиксируем зависимости..."
    source .venv_wsl/bin/activate
    pip freeze > requirements.txt
    echo "📝 Файл requirements.txt успешно обновлен."
elif [ -d "ml_env" ]; then
    echo "🔄 Найдено окружение ml_env. Фиксируем зависимости..."
    source ml_env/bin/activate
    pip freeze > requirements.txt
    echo "📝 Файл requirements.txt успешно обновлен."
else
    echo "⚠️ Виртуальное окружение не найдено в корне проекта! requirements.txt не обновлен."
fi

# Имя итогового архива
ARCHIVE_NAME="corporate_assistant_homework.tar.gz"

# 3. Архивируем проект, исключая тяжелый локальный кэш, базы и виртуальные папки
# Добавлен .env.example в список сохраняемых файлов
tar --exclude='ml_env' \
    --exclude='.venv_wsl' \
    --exclude='storage' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.vscode' \
    --exclude='.env' \
    --exclude='*.tar.gz' \
    -czf $ARCHIVE_NAME config data src main.py ingestion.py finetune.py requirements.txt prompts.yaml .env.example README.md 2>/dev/null

echo "✅ Проект успешно упакован в файл: $ARCHIVE_NAME"
echo "ℹ️ Этот архив весит всего несколько килобайт и готов к отправке преподавателю!"
