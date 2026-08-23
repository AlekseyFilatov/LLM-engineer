# 1. Создаем чистую папку для проекта в Linux
mkdir -p /home/alexfil/LLM-Training

# 2. Копируем исходники, полностью ИГНОРИРУЯ тяжелый мусор, venv и кэши
rsync -av --exclude='.venv_wsl' --exclude='ml_env' --exclude='.venv' --exclude='storage' --exclude='__pycache__' /mnt/d/LLM-Training/ /home/alexfil/LLM-Training/

# 3. Переходим в новую сверхбыструю папку
cd /home/alexfil/LLM-Training
