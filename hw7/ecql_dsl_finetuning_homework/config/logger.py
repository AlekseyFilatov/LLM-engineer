import os
import logging
from pathlib import Path

def setup_logger():
    # Создаем папку под логи внутри storage, если её нет
    log_dir = Path("./storage/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "app.log"
    
    logger = logging.getLogger("CorporateAssistant")
    logger.setLevel(logging.INFO)
    
    # Если хендлеры уже настроены, не добавляем их повторно
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Хендлер для записи в файл
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Хендлер для вывода в консоль VSCode
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()
