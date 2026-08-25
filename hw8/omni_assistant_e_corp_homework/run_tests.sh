#!/usr/bin/env bash
# Жесткий режим Bash: падать при любой ошибке
set -euo pipefail

TITLE='\033[1;36m'
NC='\033[0m'

echo -e "${TITLE}🚀 [LLMOPS] Запуск сквозного тестирования ядра E-Corp...${NC}\n"

PROJECT_ROOT="/home/alexfil/LLM-Training"
cd "$PROJECT_ROOT"

# ИСПРАВЛЕНИЕ ПУТИ: Используем абсолютный путь к Вашему реальному домашнему окружению
REAL_ENV_PYTHON="/home/alexfil/ml_env/bin/python3"

# Силовой запуск нашего программного Python-раннера
"$REAL_ENV_PYTHON" run_all_tests.py
