import time
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type

from pydantic import ValidationError
from transformers import PreTrainedTokenizerFast
from peft import PeftModel

from src.domain.monads import Result
from src.infrastructure.llm.model_loader import LocalModelTrainingLoader
from src.infrastructure.llm.trainer_factory import LocalTrainerFactory, TrainingHyperparameters
from src.infrastructure.datasets.dataset_handler import LocalDatasetHandler
from config.settings import settings
from src.domain.interfaces import IModelTrainingLoader, IDatasetHandler, ITrainerFactory


logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """Оркестрирует процесс SFT-дообучения модели:
    загрузка данных, подготовка модели, конфигурация и запуск обучения,
    сохранение результатов.
    """

    def __init__(
        self,
        model_loader: Optional[IModelTrainingLoader] = None,
        dataset_handler: Optional[IDatasetHandler] = None,
        trainer_factory: Optional[ITrainerFactory] = None,
    ) -> None:
        """Внедрение зависимостей (Dependency Injection) для изоляции и тестируемости."""
        self.model_loader = model_loader or LocalModelTrainingLoader()
        self.dataset_handler = dataset_handler or LocalDatasetHandler()
        self.trainer_factory = trainer_factory or LocalTrainerFactory()

    def execute(self, train_dataset_path: str) -> Result[Dict[str, Any]]:
        logger.info("=== [USE CASE] Инициализация infraestructura локального SFT ===")
        
        # 1. Загрузка и форматирование датасета JSONL
        logger.debug("Загрузка и форматирование датасета: %s", train_dataset_path)
        dataset_res = self.dataset_handler.format_for_sft(train_dataset_path)
        if dataset_res.is_failure():
            logger.error("Ошибка форматирования датасета: %s", dataset_res.error)
            return Result.failure(dataset_res.error, dataset_res.error_code)
        train_dataset = dataset_res.value

        # 2. Инициализация квантованной модели и LoRA-адаптера со спецтокенами
        logger.debug("Подготовка модели и токенизатора для обучения")
        model_res = self.model_loader.load_and_prepare_for_training()
        if model_res.is_failure():
            logger.error("Ошибка загрузки модели: %s", model_res.error)
            return Result.failure(model_res.error, model_res.error_code)
        model, tokenizer = model_res.value

        # 3. Валидация и конфигурация гиперпараметров из YAML через Pydantic
        logger.debug("Валидация гиперпараметров обучения")
        try:
            hyperparams = TrainingHyperparameters(**settings.training_hyperparameters)
        except ValidationError as e:
            logger.error("Ошибка валидации гиперпараметров: %s", e)
            return Result.failure(str(e), "HYPERPARAMS_VALIDATION_ERROR")
        except Exception as e:
            logger.exception("Неожиданная ошибка при валидации гиперпараметров")
            return Result.failure(f"Ошибка валидации гиперпараметров: {e}", "HYPERPARAMS_VALIDATION_ERROR")

        # 4. Создание SFTTrainer через фабрику с Chat Template маскированием
        logger.debug("Создание SFTTrainer")
        trainer_res = self.trainer_factory.create_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            config=hyperparams,
        )
        if trainer_res.is_failure():
            logger.error("Ошибка создания SFTTrainer: %s", trainer_res.error)
            return Result.failure(trainer_res.error, trainer_res.error_code)
        trainer = trainer_res.value

        # 5. Запуск тренировочного цикла на GPU ноутбука
        logger.info("Запуск обучения модели на GPU")
        start_time = time.perf_counter()
        try:
            trainer_output = trainer.train()
        except Exception as e:
            logger.exception("Критический сбой во время обучения")
            return Result.failure(f"Сбой во время обучения: {e}", "TRAINING_CRASH")
            
        runtime = time.perf_counter() - start_time
        
        # Извлечение итоговой статистики градиентного спуска
        metrics_summary = self._extract_metrics(trainer_output, runtime)
        logger.info(
            "Обучение успешно завершено. Время: %.2f сек, шаги: %d, loss: %.4f",
            metrics_summary["train_runtime_sec"],
            metrics_summary["global_step"],
            metrics_summary["train_loss"],
        )

        # 6. Безопасное сохранение весов и артефактов адаптера на жесткий диск
        output_dir = Path(hyperparams.output_dir)
        logger.info("Сохранение весов адаптера в: %s", output_dir)
        save_res = self._save_model_artifacts(trainer.model, tokenizer, output_dir)
        if save_res.is_failure():
            logger.error("Не удалось сохранить веса адаптера на диск: %s", save_res.error)
            return Result.failure(save_res.error, "WEIGHTS_SAVE_FAILED")

        # 7. Усиленная пост-очистка VRAM (Выгружаем терабайты тензоров)
        try:
            import torch
            del trainer
            del model
            gc.collect()            # Сначала чистим ссылки в Python ОЗУ
            torch.cuda.empty_cache() # Затем отдаем освобожденные блоки видеокарте
            logger.debug("Память CUDA успешно очищена после обучения.")
        except ImportError:
            pass

        logger.info("Веса успешно сохранены. Модель полностью готова к инференсу.")
        return Result.success(metrics_summary)

    @staticmethod
    def _extract_metrics(trainer_output: Any, runtime: float) -> Dict[str, Any]:
        """Извлекает финальные метрики шагов и ошибок из объекта TrainOutput Hugging Face."""
        metrics = getattr(trainer_output, "metrics", {}) if trainer_output else {}
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Фоллбэк изменен на строгий 0.0, убран ошибочный runtime_sec
        train_loss = metrics.get("train_loss") or metrics.get("loss", 0.0)
        if not isinstance(train_loss, (int, float)):
            train_loss = 0.0
            
        return {
            "train_runtime_sec": float(runtime),
            "global_step": int(getattr(trainer_output, "global_step", 0)),
            "train_loss": float(train_loss),
        }

    @staticmethod
    def _save_model_artifacts(model: Any, tokenizer: Any, output_dir: Path) -> Result[None]:
        """Изолированная стадия создания папок и записи весов LoRA-адаптера."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(output_dir))
            else:
                return Result.failure("Переданный объект модели не поддерживает save_pretrained", "INVALID_MODEL_OBJECT")          
            tokenizer.save_pretrained(str(output_dir))
            return Result.success(None)
        except Exception as e:
            return Result.failure(str(e), "WEIGHTS_SAVE_FAILED")