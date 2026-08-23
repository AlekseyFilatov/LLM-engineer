import os
import sys
import json
import random
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Корректная фиксация путей для WSL окружения
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.logger import logger
from config.settings import settings
from src.domain.monads import Result
from src.use_cases.dataset_graph import DatasetGeneratorGraph


async def main() -> None:
    """Главная точка входа для запуска асинхронного генератора датасетов ECQL."""
    logger.info("=== [GEN START] Инициализация итеративного ИИ-генератора датасетов ECQL ===")
    
    # Стартовое состояние пайплайна читается из центрального синглтона настроек settings
    initial_state: Dict[str, Any] = {
        "config": {
            "vllm_base_url": settings.VLLM_BASE_URL,
            "model_name": settings.MODEL_NAME,
            "total_target_samples": settings.TOTAL_TARGET_SAMPLES,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "max_iterations": settings.MAX_ITERATIONS,
            "prompts": {
                "ecql_generation": settings.ecql_generation_prompt
            }
        },
        "task_pool": [],
        "current_candidates": [],
        "loop_count": 0,
        "error_rail": Result.success(None)
    }

    # Запуск LangGraph оркестратора с ANN-дедупликацией
    bot_builder = DatasetGeneratorGraph(initial_state["config"])
    final_state = await bot_builder.graph.ainvoke(initial_state)

    if final_state["error_rail"].is_failure():
        logger.error("❌ Пайплайн генерации аварийно остановлен: %s", final_state["error_rail"].error)
        return

    pool: List[Dict[str, str]] = final_state["task_pool"]
    logger.info("📊 Сгенерировано чистых уникальных пар: %d", len(pool))

    # Безопасное создание директорий для артефактов
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Безопасное чтение корпоративного каталога данных для Schema Grounding
    base_path = Path(__file__).resolve().parent
    schema_path = (base_path / "./config/data_schema.yaml").resolve()
    
    if not schema_path.exists():
        logger.error("🚨 Критическая ошибка: Файл схемы данных отсутствует по пути: %s", schema_path)
        return

    try:
        with schema_path.open("r", encoding="utf-8") as f:
            catalog_data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.exception("❌ Сбой чтения YAML конфигурации схемы данных: %s", str(e))
        return

    final_dataset: List[Dict[str, str]] = []
    
    # Сборка финального датасета с инъекцией контекста схемы таблиц
    for item in pool:
        output_query: str = item["output"]
        
        # Автоматически определяем используемую сущность (например, PROJECTS)
        import re
        entity_match = re.search(r'\[([A-Z_]+)\]', output_query)
        entity_name = entity_match.group(1) if entity_match else "EMPLOYEES"
        
        # Вытаскиваем описание полей из каталога данных для Schema Grounding
        entity_info = catalog_data.get("entities", {}).get(entity_name, {})
        
        # ИСПРАВЛЕНО: Сохраняем полное человеческое описание полей для обучения сопоставлению синонимов
        fields_desc = ", ".join([f"{f} ({d})" for f, d in entity_info.get("fields", {}).items()])
        
        # Формируем инструкцию Schema Grounding (Слепок схемы вшивается в каждый пример!)
        smart_instruction = (
            f"Переведи запрос на язык ECQL компании E-Corp. "
            f"Используй доступную схему сущности [{entity_name}]. Допустимые поля: {fields_desc}."
        )

        final_dataset.append({
            "instruction": smart_instruction,
            "input": item["input"],
            "output": output_query
        })

    # Перемешиваем пул для честного случайного разбиения распределения
    random.shuffle(final_dataset)
    
    # Расчет пропорции Train / Test (80 / 20 согласно ТЗ)
    split_idx = int(len(final_dataset) * 0.8)
    train_set = final_dataset[:split_idx]
    test_set = final_dataset[split_idx:]

    # Сохраняем Обучающую выборку (Train)
    train_path = data_dir / "ecql_train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Сохраняем Тестовую выборку (Test)
    test_path = data_dir / "ecql_test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("💾 Датасет успешно дедуплицирован, разбит и сохранен на диск ноутбука!")
    logger.info(" 📝 Обучающая выборка (Train): %s (%d строк)", train_path, len(train_set))
    logger.info(" 📝 Тестовая выборка (Test): %s (%d строк)", test_path, len(test_set))


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("⚠️ Процесс генерации датасета принудительно прерван пользователем.")
