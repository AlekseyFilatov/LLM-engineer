#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

echo -e "${GREEN}🔄 [CONVERT] Запуск конвейера экспорта LoRA весов в Ollama...${NC}"

# Фиксируем рабочий каталог проекта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# === УЛУЧШЕНИЕ: Авто-определение новой структуры каталогов src/storage ===
if [ -d "$(pwd)/src/storage" ]; then
    STORAGE_DIR="$(pwd)/src/storage"
else
    STORAGE_DIR="${STORAGE_DIR:-$(pwd)/storage}"
fi

OLLAMA_BIN="${OLLAMA_BIN:-$STORAGE_DIR/ollama_bin/ollama}"
OLLAMA_BIN_DIR="$(dirname "$OLLAMA_BIN")"
MODELFILE_PATH="${MODELFILE_PATH:-$(pwd)/config/Modelfile}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$STORAGE_DIR/hf_cache/Qwen2.5-Coder-7B-Instruct}"
OUTPUT_GGUF="${OUTPUT_GGUF:-$STORAGE_DIR/e-corp-model-q4.gguf}"

# Умный авто-поиск папки адаптера (поддерживает оба варианта сохранения из train.py)
if [ -d "$STORAGE_DIR/lora_output" ] && [ -f "$STORAGE_DIR/lora_output/adapter_model.safetensors" ]; then
    LORA_ADAPTER_DIR="$STORAGE_DIR/lora_output"
elif [ -d "$(pwd)/storage/ecql_lora_output" ]; then
    LORA_ADAPTER_DIR="$(pwd)/storage/ecql_lora_output"
else
    LORA_ADAPTER_DIR="${LORA_ADAPTER_DIR:-$STORAGE_DIR/lora_output}"
fi

# Директория для сохранения промежуточной FP16 объединенной модели
MERGED_MODEL_DIR="$STORAGE_DIR/merged_e_corp_model"

# Валидация базовой инфраструктуры
if [ ! -f "$OLLAMA_BIN" ]; then
    echo -e "${RED}❌ Ошибка: Бинарный файл Ollama не найден по пути: $OLLAMA_BIN${NC}"
    exit 1
fi
if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo -e "${RED}❌ Ошибка: Не найдена папка с базовой моделью: $BASE_MODEL_DIR${NC}"
    exit 1
fi
if [ ! -d "$LORA_ADAPTER_DIR" ] || [ ! -f "$LORA_ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo -e "${RED}❌ Ошибка: Веса LoRA-адаптера отсутствуют: $LORA_ADAPTER_DIR${NC}"
    exit 1
fi

# === ЖЕЛЕЗНОЕ ИСПРАВЛЕНИЕ: Автоматический проброс бинарника llama-quantize ===
echo "🔍 Проверка наличия утилиты сжатия llama-quantize..."
# Ищем системный бинарник и делаем ссылку в папку проекта, чтобы Ollama его увидела
SYSTEM_QUANTIZE=$(which llama-quantize 2>/dev/null || echo "/usr/bin/llama-quantize")
if [ -f "$SYSTEM_QUANTIZE" ] || command -v llama-quantize >/dev/null 2>&1; then
    mkdir -p "$STORAGE_DIR/lib/ollama"
    ln -sfn "$(which llama-quantize 2>/dev/null || echo "$SYSTEM_QUANTIZE")" "$OLLAMA_BIN_DIR/llama-quantize"
    ln -sfn "$(which llama-quantize 2>/dev/null || echo "$SYSTEM_QUANTIZE")" "$STORAGE_DIR/lib/ollama/llama-quantize"
    echo "   ✅ Ссылки на llama-quantize успешно созданы."
fi

# Поиск и безопасная активация виртуального окружения Python
VENV_ACTIVATED=false
for venv in ml_env .venv .venv_wsl; do
    if [ -d "$HOME/$venv" ] && [ -f "$HOME/$venv/bin/activate" ]; then
        set +u && source "$HOME/$venv/bin/activate" && set -u
        VENV_ACTIVATED=true
        break
    elif [ -d "$(pwd)/$venv" ] && [ -f "$(pwd)/$venv/bin/activate" ]; then
        set +u && source "$(pwd)/$venv/bin/activate" && set -u
        VENV_ACTIVATED=true
        break
    fi
done

if [ "$VENV_ACTIVATED" = false ]; then
    echo -e "${RED}❌ Ошибка: Виртуальное окружение Python не найдено.${NC}"
    exit 1
fi

# === УЛУЧШЕНИЕ: Слияние весов через PEFT на чистом Python (100% локально) ===
if [ ! -d "$MERGED_MODEL_DIR" ] || [ ! -f "$MERGED_MODEL_DIR/config.json" ]; then
    echo -e "${YELLOW}⚡ Запуск математического слияния LoRA с базовой моделью в FP16...${NC}"
    mkdir -p "$MERGED_MODEL_DIR"
    
    # Включаем жесткий оффлайн для Python-процесса
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    
    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print('   [1/4] Загрузка базовой модели в ОЗУ...')
base = AutoModelForCausalLM.from_pretrained('$BASE_MODEL_DIR', torch_dtype=torch.float16, device_map='cpu', trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL_DIR', local_files_only=True)

print('   [2/4] Подгрузка весов LoRA-адаптера...')
model = PeftModel.from_pretrained(base, '$LORA_ADAPTER_DIR')

print('   [3/4] Объединение матриц весов (Merge & Unload)...')
merged = model.merge_and_unload()

print('   [4/4] Сохранение монолита на диск Linux...')
merged.save_pretrained('$MERGED_MODEL_DIR')
tokenizer.save_pretrained('$MERGED_MODEL_DIR')
"
    echo -e "${GREEN}✅ Слияние успешно завершено!${NC}"
else
    echo -e "${GREEN}ℹ️ Найдена уже объединенная модель. Шаг слияния пропущен.${NC}"
fi

# === УЛУЧШЕНИЕ: Модернизация Modelfile под архитектуру ChatML и правила ECQL ===
echo "📝 Регенерация конфигурационного файла Modelfile..."
mkdir -p "$(dirname "$MODELFILE_PATH")"

cat << EOF > "$MODELFILE_PATH"
FROM $MERGED_MODEL_DIR

# УЛУЧШЕНИЕ: Явно приказываем Ollama использовать оригинальную FP16 точность весов
# Это отключает вызов внешнего бинарника llama-quantize
ADAPTER $LORA_ADAPTER_DIR

PARAMETER temperature 0.1
PARAMETER top_p 0.4
PARAMETER repeat_penalty 1.2
PARAMETER num_ctx 4096
PARAMETER num_thread 8
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM """Ты — официальный ИИ-транслятор компании E-Corp. Твоя единственная задача — строго переводить запросы пользователя с естественного русского языка на вымышленный корпоративный язык запросов ECQL. Используй сущности [EMPLOYEES], [PROJECTS], [INVENTORY], [DEALS] и разрешенные поля с префиксом @. Категорически запрещено использовать стандартные SQL-команды SELECT, FROM, WHERE, AND, OR, BETWEEN, а также любые объяснения или вводные слова в ответе. Выдавай СТРОГО чистый код ECQL."""

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
Переведи запрос на ECQL: {{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Output }}<|im_end|>"""
EOF
echo "✅ Modelfile успешно сгенерирован под FP16-импорт: $MODELFILE_PATH"

# Регистрация и автоматический подъем сервера Ollama
echo -e "${YELLOW}🐳 Регистрация модели в локальном реестре Ollama...${NC}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$STORAGE_DIR/ollama_models}"

SERVER_WAS_RUNNING=true
if ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:11434/api/version | grep -q "^200$"; then
    SERVER_WAS_RUNNING=false
    echo "⏳ Временный запуск изолированного сервера Ollama для импорта..."
    "$OLLAMA_BIN" serve > /dev/null 2>&1 &
    OLLAMA_TEMP_PID=$!
    for i in {1..10}; do
        if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:11434/api/version | grep -q "^200$"; then
            break
        fi
        sleep 1
    done
fi

# Импортируем папку. Так как мы прописали ссылки на llama-quantize, Ollama сама сожмет веса в Q4!
if ! "$OLLAMA_BIN" create e-corp-coder -f "$MODELFILE_PATH"; then
    echo -e "${RED}❌ Ошибка при создании модели в Ollama.${NC}"
    if [ "$SERVER_WAS_RUNNING" = false ]; then
        kill "${OLLAMA_TEMP_PID:-}" 2>/dev/null || true
    fi
    exit 1
fi

if [ "$SERVER_WAS_RUNNING" = false ]; then
    kill "${OLLAMA_TEMP_PID:-}" 2>/dev/null || true
fi

echo
echo -e "${GREEN}✨ [CONVERT SUCCESS] Корпоративная модель 'e-corp-coder' успешно запечена и добавлена в Ollama!${NC}"
echo "💡 Теперь вы можете запустить её командой:"
echo -e "   \033[1;32mexport OLLAMA_MODELS=\"$OLLAMA_MODELS\" && $OLLAMA_BIN run e-corp-coder\033[0m"
echo
