import os
import sys
import gc
import torch
import asyncio
from typing import Dict, Any
import logging

# Корректная фиксация путей для WSL окружения
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.logger import logger
from src.use_cases.train_model import TrainModelUseCase
from src.domain.monads import Result

# Подключаем ваш проверенный синглтон настроек
from config.settings import settings


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def run_fine_tuning_pipeline() -> None:
    """Управляющий скрипт инициализации оборудования, очистки VRAM и запуска SFT."""
    logger.info("=== [TRAIN START] Инициализация локальной инфраструктуры SFT ===")
    
    # 1. Аппаратный защищенный фильтр (Защита вычислений на ноутбуке)
    if not torch.cuda.is_available():
        logger.error("🚨 Критическая ошибка: Дообучение невозможно. Графический ускоритель GPU (CUDA) не зафиксирован.")
        return

    # Дополнительное улучшение: Считываем ТТХ вашей RTX 5070 Ti для верификации ресурсов
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"💻 Аппаратный контекст: {gpu_name} (Всего VRAM: {total_vram:.2f} GB)")

    # 2. Упреждающий сброс VRAM (Освобождение памяти под SFTTrainer)
    logger.debug("🔄 Сброс и изоляция видеопамяти ноутбука...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # Метрика свободной памяти после cleanup.sh
    free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    logger.info(f"✅ Кэш видеопамяти очищен. Доступно для обучения: {free_vram:.2f} GB VRAM.")

    # 3. Конфигурация путей к данным
    # УЛУЧШЕНИЕ: Путь строится динамически (из папки data/, которую наполнил split_dataset.py)
    train_dataset_file: str = "./data/ecql_train.jsonl"
    
    if not torch.os.path.exists(train_dataset_file):
        logger.error(f"🚨 Ошибка старта: Файл обучающей выборки отсутствует по пути: {train_dataset_file}")
        logger.info("💡 Пожалуйста, запустите сначала скрипт split_dataset.py для выгрузки данных из ChromaDB.")
        return

    # 4. Запуск Use Case в изолированном системном потоке
    # Внедрение зависимостей произойдет автоматически внутри конструктора UseCase
    use_case = TrainModelUseCase()
    logger.info("🔥 Передача управления в оркестратор TrainModelUseCase...")
    
    # Обучение — процесс синхронный и тяжелый, asyncio.to_thread уберет его 
    # из главного event loop, защищая интерфейс VSCode от зависания
    result: Result[Dict[str, Any]] = await asyncio.to_thread(use_case.execute, train_dataset_file)

    # 5. Обработка результатов на рельсе безопасности
    print("\n" + "=" * 70)
    logger.info("🏁 ИТОГ ВЫПОЛНЕНИЯ FINE-TUNING В VSCODE:")
    print("=" * 70)
    
    if result.is_failure():
        logger.error("❌ Пайплайн обучения прерван [%s]: %s", result.error_code, result.error)
    else:
        stats: Dict[str, Any] = result.value
        logger.info("🎉 Успех! Локальный LoRA-адаптер под синтаксис ECQL успешно собран.")
        logger.info("📉 Финальная ошибка модели (Loss): %.4f", stats.get("train_loss", 0.0))
        logger.info("⏱️ Чистое время работы градиентного спуска: %.2f сек", stats.get("train_runtime_sec", 0.0))

    # 6. Пост-очистка (Освобождаем RTX 5070 Ti сразу после завершения шага)
    logger.debug("🧹 Пост-тренировочная зачистка контекстов...")
    del use_case
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✅ Память GPU успешно возвращена операционной системе.")

if __name__ == "__main__":
    try:
        asyncio.run(run_fine_tuning_pipeline())
    except KeyboardInterrupt:
        logger.warning("⚠️ Процесс обучения принудительно прерван пользователем (Ctrl+C).")
    except Exception as e:
        logger.exception("❌ Критический крах верхнего уровня ядра train.py: %s", str(e))
