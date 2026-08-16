#!/bin/bash

echo "📥 [OLLAMA SETUP] Инициализация локальной ИИ-инфраструктуры..."
#!/usr/bin/env bash
set -e # Остановить скрипт при любой ошибке

# 1. Создаем изолированные папки
mkdir -p ./storage/ollama_bin
mkdir -p ./storage/ollama_models
mkdir -p ./storage/tmp

echo "📥 Определение архитектуры системы..."
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
else
    echo "❌ Неподдерживаемая архитектура: $ARCH"
    exit 1
fi
echo "✅ Архитектура: $ARCH"

echo "📥 Скачивание и локальная установка Ollama через официальный конвейер архивов..."

# 1. Динамически вычисляем самую последнюю стабильную версию
echo "🔍 Запрос актуальной версии с GitHub..."
OLLAMA_VERSION=$(curl -sI https://github.com | grep -i "location:" | awk -F/ '{print $NF}' | tr -d '\r\n')

if [ -z "$OLLAMA_VERSION" ]; then
    echo "⚠️ Не удалось определить версию автоматически. Используем фоллбэк v0.32.6"
    OLLAMA_VERSION="v0.32.6"
fi
echo "📦 Актуальная версия для скачивания: $OLLAMA_VERSION"

# 2. Формируем правильный URL к официальному архиву (как в install.sh)
# Пример: https://github.com
OLLAMA_URL="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-${ARCH}.tar.zst"
TMP_ARCHIVE="./storage/tmp/ollama.tar.zst"

echo "📥 Скачивание официального архива..."
if ! curl -L -f --connect-timeout 15 --retry 5 --retry-delay 3 -o "$TMP_ARCHIVE" "$OLLAMA_URL"; then
    echo "❌ Ошибка: Не удалось скачать архив Ollama. Проверьте сеть или URL: $OLLAMA_URL"
    exit 1
fi

# 3. Распаковываем архив во временную папку и забираем только сам бинарник
echo "📦 Распаковка и извлечение локального компонента..."

# Создаем временную структуру для распаковки
mkdir -p ./storage/tmp/extracted

echo "✅ Выверенный URL для скачивания: $OLLAMA_URL"
echo "✅ Архив успешно сохранен в: $TMP_ARCHIVE"

# 3. Распаковываем архив во временную папку и забираем только сам бинарник
echo "📦 Распаковка и извлечение локального компонента..."

# Создаем временную структуру для распаковки
mkdir -p ./storage/tmp/extracted

# Проверяем наличие zstd в WSL. 
# Если его нет — автоматически устанавливаем через пакетный менеджер.
if ! command -v zstd >/dev/null 2>&1; then
    echo "⚠️ Утилита zstd не найдена. Пытаемся установить пакет в WSL..."
    if command -v apt-get >/dev/null 2>&1; then
        # Обновляем списки пакетов и ставим zstd тихо (-y)
        sudo apt-get update -qq && sudo apt-get install -y zstd -qq
        echo "✅ Пакет zstd успешно установлен в систему."
    else
        echo "❌ Ошибка: В системе отсутствует архиватор zstd, и менеджер пакетов apt не найден."
        echo "   Пожалуйста, установите zstd вручную в вашей ОС."
        exit 1
    fi
fi

# Распаковываем tar.zst архив. 
# Из-за 'set -euo pipefail' мы убрали хрупкие конструкции '||' и вызываем команду напрямую.
tar --use-compress-program=zstd -xf "$TMP_ARCHIVE" -C ./storage/tmp/extracted

# В официальном архиве сам бинарник лежит строго по пути bin/ollama
if [ -f "./storage/tmp/extracted/bin/ollama" ]; then
    mv "./storage/tmp/extracted/bin/ollama" ./storage/ollama_bin/ollama
    
    # Забираем официальные либы поддержки GPU (CUDA), которые шли в архиве
    if [ -d "./storage/tmp/extracted/lib/ollama" ]; then
        mkdir -p ./storage/ollama_bin/lib
        # Используем cp -pf, чтобы сохранить права доступа к библиотекам
        cp -rpf ./storage/tmp/extracted/lib/ollama/* ./storage/ollama_bin/lib/

        # Выносим сам llama-server в корень ollama_bin
        cp -pf ./storage/ollama_bin/lib/llama-server ./storage/ollama_bin/llama-server
        chmod +x ./storage/ollama_bin/llama-server

        cp -pf ./storage/ollama_bin/lib/*.so ./storage/ollama_bin/ 2>/dev/null || true
        
        echo "⚡ Официальные библиотеки ускорения GPU успешно импортированы."

        chmod +x ./storage/ollama_bin/llama-server
    fi
else
    echo "❌ Ошибка: Структура архива изменилась или распаковка прошла некорректно. bin/ollama не найден."
    exit 1
fi

# 4. Выдаем права и подчищаем за собой тяжелый мусор
chmod +x ./storage/ollama_bin/ollama
rm -rf ./storage/tmp

echo -e "\n✅ [SUCCESS] Официальный стабильный релиз Ollama ${OLLAMA_VERSION} успешно развернут локально!"
echo "📂 Путь к изолированному исполняемому файлу: ./storage/ollama_bin/ollama"

# 4. Программный манифест: Скачивание модели Qwen
echo "🔥 Запуск временного сервера Ollama для скачивания модели..."

# Задаем переменную OLLAMA_MODELS, заставляя Ollama качать веса в папку проекта
export OLLAMA_MODELS="./storage/ollama_models"

# Проверяем, не занят ли порт системной службой
LOCAL_SERVER_RUNNING=false
if curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
    echo "⚠️ Порт 11434 уже занят запущенной Ollama. Используем существующий сервер."
    LOCAL_SERVER_RUNNING=true
else
    # Запускаем локальный сервер в фоновом режиме
    ./storage/ollama_bin/ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    
    # Динамическое ожидание старта сервера (максимум 20 секунд) вместо sleep 3
    echo "⏳ Ожидание инициализации API сервера..."
    SUCCESS=false
    for i in {1..20}; do
        if curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
            SUCCESS=true
            break
        fi
        sleep 1
    done

    if [ "$SUCCESS" = false ]; then
        echo "❌ Ошибка: Сервер Ollama не ответил по API за 20 секунд."
        kill $OLLAMA_PID 2>/dev/null || true
        exit 1
    fi
fi

echo "📥 Скачивание модели qwen2.5-coder:7b в каталог приложения (~4.7 GB)..."
# Запускаем скачивание. Полный путь к бинарнику гарантирует использование нашей версии
./storage/ollama_bin/ollama pull qwen2.5-coder:7b

# Извлекаем список скачанных моделей для верификации
echo "📊 Список локально установленных моделей в проекте:"
./storage/ollama_bin/ollama list

# Изящно тушим сервер ТОЛЬКО если мы сами его запустили
if [ "$LOCAL_SERVER_RUNNING" = false ]; then
    echo "⏹️ Остановка временного сервера Ollama..."
    kill $OLLAMA_PID
    wait $OLLAMA_PID 2>/dev/null || true
fi

echo "✅ [SUCCESS] Локальная ИИ-инфраструктура полностью собрана внутри папки storage/!"
