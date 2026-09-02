# 1. Скачиваем официальный скомпилированный релиз Triton v2.54 с поддержкой vLLM бэкенда
wget -q --show-progress https://github.com/triton-inference-server/server/releases/download/v2.54.0/tritonserver2.54.0-vllm.tar.gz

# 2. Безопасно распаковываем бинарники в системный путь /usr/local
sudo tar -xzf tritonserver2.54.0-vllm.tar.gz -C /usr/local/ --strip-components=1

# 3. Принудительно изолируем видеокарту RTX 5070 Ti через CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# 4. Промышленный запуск сервера с верными флагами и вербальным логированием для отчета
tritonserver \
  --model-repository=/home/user/triton/models \
  --allow-http=true \
  --allow-grpc=true \
  --allow-metrics=true \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --log-verbose=1



#!/bin/bash

# Останавливаем скрипт при любой ошибке
set -e

echo "🚀 Проверка и подготовка инфраструктуры Triton..."

# Определяем абсолютный путь к репозиторию моделей
REPO_DIR="$(pwd)/triton_repository"

# Создаем базовую директорию, если она не существует
if [ ! -d "$REPO_DIR" ]; then
    echo "📁 Создаю директорию $REPO_DIR"
    mkdir -p "$REPO_DIR"
fi

echo "🐳 Запуск контейнера NVIDIA Triton Server..."

# Облегченный образ Triton только с поддержкой ONNX и Python бэкендов
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  tritonserver:25.01-py3-min \
  tritonserver --model-repository=/models