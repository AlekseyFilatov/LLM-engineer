#!/usr/bin/env bash
set -euo pipefail

### Цвета для вывода

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "𝐺𝑅𝐸𝐸𝑁[𝑇𝑅𝐼𝑇𝑂𝑁𝑆𝐸𝑇𝑈𝑃]{NC} $1"; }
log_warn() { echo -e "𝑌𝐸𝐿𝐿𝑂𝑊[𝑊𝐴𝑅𝑁]{NC} $1"; }
log_err()  { echo -e "𝑅𝐸𝐷[𝐸𝑅𝑅𝑂𝑅]{NC} $1"; }

log_info "Развертывание высоконагруженного ML-окружения E-Corp..."

### Защита от CRLF переносов строк (если файл редактировался в Windows)

if [[ $(file "$0" 2>/dev/null) == *"CRLF"* ]]; then
log_warn "Обнаружены Windows-переносы строк (CRLF). Автоматически очищаю..."
sed -i 's/\r//' "$0"
fi

### Проверка наличия python3

if ! command -v python3 &> /dev/null; then
log_err "В системе не найден python3."
echo "💡 Выполните: sudo apt update && sudo apt install -y python3"
exit 1
fi

### ==========================================

### ШАГ 1: Обнаружение или создание окружения

### ==========================================

ENV_DIR=""

if [ -n "${VIRTUAL_ENV:-}" ]; then
ENV_DIR="$VIRTUAL_ENV"
log_info "✅ Зафиксировано active окружение в терминале: (basename "ENV_DIR")"
elif [ -d ".venv" ]; then
ENV_DIR="$(pwd)/.venv"
log_info "🔄 Найдена папка .venv. Используем её..."
elif [ -d "ml_env" ]; then
ENV_DIR="$(pwd)/ml_env"
log_info "🔄 Найдена папка ml_env. Используем её..."
else
log_info "🔄 Локальное окружение не найдено. Создаем чистый .venv..."
if ! python3 -m venv .venv 2>/dev/null; then
log_err "Не удалось создать виртуальное окружение."
echo "💡 Проверьте установку пакета python3-venv: sudo apt install -y python3-venv"
exit 1
fi
ENV_DIR="$(pwd)/.venv"
fi

ENV_PIP="$ENV_DIR/bin/pip"
ENV_PYTHON="$ENV_DIR/bin/python"

log_info "Обновление менеджера пакетов pip..."
"$ENV_PIP" install --upgrade pip -q

### ==========================================

### ШАГ 2: Установка PyTorch под тип железа

### ==========================================

log_info "Определение конфигурации GPU..."

if ! command -v nvidia-smi &> /dev/null; then
log_warn "nvidia-smi не найден. Установка стабильной CPU-версии PyTorch..."
"$ENV_PIP" install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu -q
else
log_info "✅ Обнаружена видеокарта NVIDIA. Установка PyTorch с поддержкой CUDA 12.1..."
"$ENV_PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi

### ==========================================

### ШАГ 3: Установка зависимостей проекта

### ==========================================

log_info "Установка зависимостей (Triton Client, ONNX Runtime GPU, Tokenizers)..."

"$ENV_PIP" install \
    tritonclient[grpc,http] \
    "numpy>=1.24.0" \
    pillow \
    requests \
    "transformers>=4.45.0" \
    tokenizers \
    sentencepiece \
    onnx \
    onnxruntime-gpu \
    "optimum[onnxruntime]" \
    scikit-learn \
    pydantic \
    pyyaml \
    tqdm \
    accelerate \
    protobuf==4.25.3 \
    -q

log_info "✅ Все базовые и вычислительные библиотеки успешно установлены!"


### ==========================================

### ШАГ 4: Фиксация и очистка

### ==========================================

log_info "Фиксация зависимостей в requirements.txt..."
"$ENV_PIP" freeze --local > requirements.txt

log_info "Очистка временного кэша pip..."
"$ENV_PIP" cache purge -q || true
log_info "Кэш pip очищен."

### ==========================================

### ШАГ 5: Инициализация структуры каталогов Triton

### ==========================================

### ==========================================
### ШАГ 5: Инициализация структуры каталогов Triton
### ==========================================
log_info "Инициализация структуры каталогов triton_repository..."

"$ENV_PYTHON" - << 'EOF'
from pathlib import Path
import os
import sys

try:
    from config.settings import settings
    project_root = Path(settings.PROJECT_ROOT)
except ImportError:
    project_root = Path.cwd()

triton_repo = project_root / 'triton_repository'
models = ['text_classifier', 'text_embedder', 'text_generator', 'pipeline_orchestrator']

try:
    for model in models:
        # 1. Безопасно создаем дерево директорий для версии
        model_ver_dir = triton_repo / model / '1'
        model_ver_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Проверяем config.pbtxt
        config_file = triton_repo / model / 'config.pbtxt'
        
        # Защита: Если config.pbtxt по ошибке создан как папка — удаляем её
        if config_file.is_dir():
            print(f"   ⚠️ Обнаружена ошибочная директория {config_file.name}, исправляю на файл...")
            config_file.rmdir()
            
        # Создаем пустой файл конфигурации, только если его еще нет
        if not config_file.exists():
            config_file.touch()
            print(f"   📝 Создан пустой конфиг-заглушка: {model}/config.pbtxt")
        else:
            print(f"   ✅ Каталог и конфиг для [{model}] уже существуют. Пропускаю.")

    print(f"   \033[1;32m✅ Вся файловая структура успешно проверена в: {triton_repo}\033[0m")

except Exception as e:
    print(f"   ❌ Ошибка при инициализации каталогов: {str(e)}", file=sys.stderr)
    sys.exit(1)
EOF

### ==========================================

### ШАГ 6: Верификация аппаратного ускорения

### ==========================================

log_info "Верификация аппаратного ускорения..."

"$ENV_PYTHON" - << 'EOF'
import torch
import sys

cuda_status = "❌"
if torch.cuda.is_available():
cuda_status = "✅"
device_name = torch.cuda.get_device_name(0)
print(f"   {cuda_status} CUDA готова! Движок PyTorch видит карту: {device_name}")
else:
print(f"   {cuda_status} CUDA недоступна для PyTorch! Проверьте проброс GPU в WSL или драйверы.")

try:
import tritonclient.grpc as grpcclient
print("   ✅ gRPC Клиент Triton успешно импортирован.")
except ImportError as e:
print(f"   ❌ Ошибка импорта клиента Triton: {e}")
sys.exit(1)
EOF

log_info "🎉 Настройка завершена! Окружение полностью готово к High-load тестам."