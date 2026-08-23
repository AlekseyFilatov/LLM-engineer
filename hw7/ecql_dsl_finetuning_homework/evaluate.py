import os
import sys
import gc
import torch
import asyncio

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.logger import logger
from src.use_cases.evaluate_model import EvaluateModelUseCase

async def run_evaluation_pipeline():
    print("=== Инициализация проверочного стенда (Evaluation) ===")
    
    if not torch.cuda.is_available():
        logger.error("Критическая ошибка: Тестирование на GPU невозможно.")
        sys.exit(1)

    gc.collect()
    torch.cuda.empty_cache()

    test_dataset_file = "./data/ecql_test.jsonl"
    use_case = EvaluateModelUseCase()
    
    # ИСПРАВЛЕНИЕ: Передаем batch_size=1, чтобы полностью убрать пустые генерации на GPU
    result = await asyncio.to_thread(use_case.execute, test_dataset_file, 1)

    print("\n" + "="*70)
    print("🏁 ИТОГОВЫЕ МЕТРИКИ ВЕРИФИКАЦИИ DSL МОДЕЛИ (WSL/VSCODE):")
    print("="*70)
    
    if result.is_failure():
        logger.error(f"❌ Валидация прервана [{result.error_code}]: {result.error}")
        return

    # Извлекаем финальный словарь отчета
    report = result.value
    errors = report.get("errors", [])

    # Безопасный вывод реестра дефектов
    print(f"\n🕵️ РЕЕСТР ВСЕХ ОТБРАКОВАННЫХ ВАРИАНТОВ (ВСЕГО: {len(errors)}):", flush=True)
    if errors:
        for i, err in enumerate(errors):
            print(f"\nДефект #{i+1}:", flush=True)
            print(f" ❓ Запрос:       '{err.get('input')}'", flush=True)
            print(f" 🟢 Ожидалось:     {err.get('expected')}", flush=True)
            print(f" 🔴 Сгенерировано: {err.get('generated')}", flush=True)
            print(f" ⚠️ Причина сбоя:  {err.get('reason')}", flush=True)
            print("-" * 60, flush=True)
    else:
        print("🎉 Модель не допустила ни одной ошибки на всем пуле тестов!", flush=True)
    
    print("\n" + "="*70, flush=True)
    
    # Итоговые метрики из Use Case (Защита от NameError)
    logger.info(f"📊 Всего протестировано примеров: {report.get('total', 0)}")
    logger.info(f"📊 Синтаксическая точность (Syntax Accuracy): {report.get('syntax_accuracy', 0.0):.1f}%")
    logger.info(f"📊 Логическая точность (Exact Match Accuracy): {report.get('exact_match_accuracy', 0.0):.1f}%")
    logger.info(f"📊 Доля SQL-галлюцинаций (SQL Hallucination Rate): {report.get('sql_hallucination_rate', 0.0):.1f}%")
    logger.info(f"📈 Структурный F1-Macro Скоростной балл: {report.get('f1_macro', 0.0):.1f}%")
    logger.info(f"📈 Precision / Recall (Macro): {report.get('precision_macro', 0.0):.1f}% / {report.get('recall_macro', 0.0):.1f}%")

if __name__ == "__main__":
    asyncio.run(run_evaluation_pipeline())
