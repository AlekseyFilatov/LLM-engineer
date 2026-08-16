import logging
from pathlib import Path
from typing import Any, Set, Final
from datasets import load_dataset

from src.domain.monads import Result
from config.logger import logger

logger = logging.getLogger(__name__)


import logging
from pathlib import Path
from typing import Any, Dict, Set, Final

# Явные импорты необходимых ML-компонентов
from datasets import load_dataset, DatasetDict

# Предполагаем, что класс Result импортирован корректно
# from your_system import Result

logger = logging.getLogger(__name__)

class LocalDatasetHandler:
    """Инфраструктурный сервис безопасной загрузки и верификации корпоративных JSONL датасетов."""

    @staticmethod
    def format_for_sft(train_dataset_path: str) -> Result[DatasetDict]:
        """Атомарно считывает Train и Test выборки с диска ноутбука и валидирует их контракты."""
        logger.info("📥 [DATASET HANDLER] Инициализация загрузки тренировочной экосистемы...")
        
        train_path = Path(train_dataset_path).resolve()
        # Автоматически вычисляем путь к тестовой выборке в той же папке data/
        test_path = (train_path.parent / "ecql_test.jsonl").resolve()

        # Проверяем физическое наличие обоих файлов на диске ext4
        if not train_path.exists():
            logger.error("❌ [DATASET HANDLER] Обучающий файл не зафиксирован на диске: %s", train_path)
            return Result.failure(f"Файл ecql_train.jsonl не обнаружен: {train_path}", "TRAIN_FILE_NOT_FOUND")
            
        if not test_path.exists():
            logger.error("❌ [DATASET HANDLER] Валидационный файл не зафиксирован на диске: %s", test_path)
            return Result.failure(f"Файл ecql_test.jsonl не обнаружен: {test_path}", "TEST_FILE_NOT_FOUND")
            
        try:
            # Загружаем ОДНОВРЕМЕННО обе выборки в единую структуру DatasetDict (100% Offline)
            data_files = {
                "train": str(train_path),
                "test": str(test_path)
            }
            
            dataset_dict = load_dataset(
                "json", 
                data_files=data_files,
                verification_mode="no_checks"  # Запрещает скачивать скрипты валидации из AWS S3
            )
            
            # 1. Гвардейская защита от пустых выборок
            if len(dataset_dict["train"]) == 0 or len(dataset_dict["test"]) == 0:
                logger.error("❌ [DATASET HANDLER] Одна из выборок содержит 0 строк. Сбой разделения.")
                return Result.failure("Файлы выборок пусты. Перезапустите split_dataset.py", "DATASET_EMPTY_ERROR")
            
            # 2. Комплексная верификация контракта полей и типов данных
            required_keys: Final[Set[str]] = {"instruction", "input", "output"}
            
            for split_name in ["train", "test"]:
                sample = dataset_dict[split_name][0]
                
                # Проверяем состав ключей структуры, сгенерированной из ChromaDB
                if not required_keys.issubset(sample.keys()):
                    logger.error("❌ [DATASET HANDLER] Поврежден контракт полей в split: %s", split_name)
                    return Result.failure(
                        f"Отсутствуют обязательные поля в {split_name}: {required_keys}", 
                        "INVALID_DATASET_STRUCTURE"
                    )
                
                # Дополнительное улучшение: Стрикт-контроль типов (Защита от падения токенизатора)
                if not isinstance(sample["input"], str) or not isinstance(sample["output"], str):
                    logger.error("❌ [DATASET HANDLER] Обнаружено нетекстовое значение в полях данных!")
                    return Result.failure("Поля input и output должны быть строго строкового типа (str)", "DATASET_TYPE_MISMATCH")

            logger.info("✅ [DATASET HANDLER] Структура выборок успешно верифицирована.")
            logger.info("   [INFO] Обучение (Train): %d пар | Валидация (Test): %d пар", 
                        len(dataset_dict["train"]), len(dataset_dict["test"]))
                        
            return Result.success(dataset_dict)
            
        except Exception as e:
            logger.exception("❌ [DATASET HANDLER] Критический сбой при парсинге JSONL структуры: %s", str(e))
            return Result.failure(f"Не удалось распарсить и загрузить JSONL датасет: {str(e)}", "DATASET_PARSE_FAILED")
