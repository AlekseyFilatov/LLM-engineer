#!/usr/bin/env bash
set -euo pipefail # Более строгий режим: ошибка при использовании неопределенных переменных

# --- Конфигурация ---
BASE_DIR="$(pwd)"
DIST_DIR="${BASE_DIR}/triton_dist"
ARCHIVE_NAME="v2.71.0-ubuntu2404.backends.tar.gz"
ARCHIVE_PATH="${BASE_DIR}/${ARCHIVE_NAME}"
# ВАЖНО: Полная ссылка на релиз
ARCHIVE_URL="https://github.com/triton-inference-server/server/releases/download/v2.71.0/${ARCHIVE_NAME}"

echo "📥 [1/4] Скачивание официальных бинарников ядра Triton Server v2.71.0..."
if [ ! -f "${ARCHIVE_PATH}" ]; then
    wget -q --show-progress -O "${ARCHIVE_PATH}" "${ARCHIVE_URL}"
else
    echo "⚠️ Архив уже существует, пропускаем скачивание."
fi

echo "🔄 [2/4] Подготовка изолированной папки и распаковка..."
mkdir -p "${DIST_DIR}"

# Распаковываем. Флаг --strip-components=1 убирает лишнюю корневую папку, если она есть в архиве.
# Это делает скрипт устойчивым к разным форматам упаковки.
tar -xzf "${ARCHIVE_PATH}" -C "${DIST_DIR}" --strip-components=1

# Проверка структуры после распаковки
if [ ! -d "${DIST_DIR}/bin" ] || [ ! -d "${DIST_DIR}/lib" ]; then
    echo "❌ Ошибка структуры архива: ожидались папки bin и lib в корне распаковки."
    ls -la "${DIST_DIR}"
    exit 1
fi

echo "📂 [3/4] Безопасный перенос файлов в системные директории /usr/local..."
sudo cp -r "${DIST_DIR}/bin/"* /usr/local/bin/
sudo cp -r "${DIST_DIR}/lib/"* /usr/local/lib/

sudo mkdir -p /usr/local/backends
sudo cp -r "${DIST_DIR}/backends/"* /usr/local/backends/

# Обновляем кэш динамических библиотек
sudo ldconfig

# Гарантируем права на выполнение бинарного файла
sudo chmod +x /usr/local/bin/tritonserver

echo "📊 [4/4] Верификация установки..."
tritonserver --version

# Простая проверка видимости GPU (опционально, покажет ошибку, если нет драйверов)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ Драйверы NVIDIA найдены. Архитектура GPU будет определена при реальном запуске сервера."
else
    echo "⚠️ Внимание: nvidia-smi не найден. Убедитесь, что драйверы NVIDIA установлены в WSL."
fi

echo "🧹 Очистка временных файлов..."
rm -rf "${ARCHIVE_PATH}" "${DIST_DIR}"

echo "🎉 [SUCCESS] Triton Server v2.71.0 успешно развернут!"
