import os
import sys
import torch
import gc
import asyncio
import chromadb

# Добавляем корневую папку в пути для явного импорта пакетов src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import settings
from config.logger import logger
from src.domain.monads import Result
from src.infrastructure.vector_stores.db_optimizer import ChromaDBOptimizer
from src.infrastructure.vector_stores import CorporateHybridSearchStore
from src.infrastructure.llm import LocalLLMClient
from src.use_cases.rag_graph import CorporateRAGGraph

async def main():
    print("="*70)
    print("⚙️ СТАДИЯ 1: ПОДГОТОВКА И ОПТИМИЗАЦИЯ ОКРУЖЕНИЯ")
    print("="*70)

    # 1. Освобождаем VRAM ноутбука от мусора прошлых запусков
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("✅ Кэш VRAM успешно сброшен.")

    # 2. Оптимизируем дисковую базу ChromaDB перед запуском
    optimizer = ChromaDBOptimizer(db_path=settings.CHROMA_DB_DIR)
    opt_res = optimizer.optimize_and_validate()
    if opt_res.is_failure():
        print(f"⚠️ Предупреждение оптимизатора: {opt_res.error}")
    else:
        print(f"✅ {opt_res.value}")

    print("\n" + "="*70)
    print("🤖 СТАДИЯ 2: ИНИЦИАЛИЗАЦИЯ ИИ-ЯДРА")
    print("="*70)
   
    # 1. Инициализируем локальную базу ChromaDB (путь берется из .env/settings)
    print("📁 [ORCHESTRATOR] Инициализация локального хранилища документов ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = chroma_client.get_or_create_collection(name="corporate_knowledge")

    # 2. Инициализируем компоненты слоя Инфраструктуры
    vector_store = CorporateHybridSearchStore(chroma_collection=collection)
    
    print("\n🔥 [ORCHESTRATOR] Запуск локального ИИ-ядра Qwen2.5...")
    llm_core = LocalLLMClient()

    # 3. Собираем Сценарий Суперагента (Use Case) через Dependency Injection
    rag_app = CorporateRAGGraph(vector_store=vector_store, llm=llm_core)

    # =====================================================================
    # БЛОК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ ИИ-АГЕНТА (5 РАЗНЫХ СЦЕНАРИЕВ)
    # =====================================================================
    
    # Список тестовых вопросов разных типов для проверки устойчивости RAG
    test_cases = [
        {
            "id": 1,
            "type": "ПРЯМОЙ ВОПРОС (Проверка точности одной страницы)",
            "query": "За сколько дней нужно писать заявление на отпуск?"
        },
        {
            "id": 2,
            "type": "ВОПРОС С АГРЕГАЦИЕЙ (Объединение информации из двух разных файлов)",
            "query": "Каковы сроки подачи заявления на отпуск и когда мне должны выплатить деньги?"
        },
        {
            "id": 3,
            "type": "ПРОВОКАЦИЯ / OUT-OF-DOMAIN (Проверка защиты от галлюцинаций)",
            "query": "Каков размер штрафа за опоздание на работу на 15 минут согласно регламенту компании?"
        },
        {
            "id": 4,
            "type": "ТЕСТ НА ДЛИНУ ОТВЕТА (Теряет ли модель ссылки при длинном тексте)",
            "query": "Подробно и развернуто опиши все правила, регламенты и финансовые выплаты, связанные с отпусками в нашей организации."
        },
        {
            "id": 5,
            "type": "ВЛИЯНИЕ СЕМАНТИЧЕСКОГО ЧАНКА (Точность определения темы/заголовка)",
            "query": "В каком месяце генеральный директор должен подписать общий график отпусков?"
        }
    ]
    
    for case in test_cases:
        print("\n" + "="*70)
        logger.info(f"🚀 ТЕСТ №{case['id']}: {case['type']}")
        print("="*70)
        
        # Конструируем изолированное стартовое состояние для каждого теста
        initial_state = {
            "question": case["query"],
            "intent": "", 
            "retrieved_docs": [], 
            "final_answer": None, # Сюда прилетит Pydantic-объект CorporateAnswerSchema
            "revision_count": 0, 
            "metrics": {"is_valid": True, "critic_feedback": ""},
            "status": "Старт", 
            "error_rail": opt_res # Передаем статус инициализации (Result) на рельсу безопасности
        }

        # Запускаем асинхронное выполнение графа
        output = await rag_app.graph.ainvoke(initial_state)

        # Формируем красивый структурированный отчет по тесту
        if output.get("error_rail") and output["error_rail"].is_failure():
            logger.error(
                f"❌ РЕЗУЛЬТАТ ТЕСТА №{case['id']}: FAILURE\n"
                f"Причина аварийного останова графа: {output['error_rail'].error}\n"
            )
        else:
            ai_response = output.get("final_answer")
            
            if ai_response:
                log_message = (
                    f"✅ РЕЗУЛЬТАТ ТЕСТА №{case['id']}: SUCCESS!\n"
                    f"❓ Вопрос пользователя: {case['query']}\n"
                    f"🤖 Текст ответа ИИ:\n{ai_response.text_answer}\n\n"
                    f"📊 Уровень уверенности ИИ (Confidence): {ai_response.confidence_score}\n"
                    f"⚠️ Требуется ручная проверка человеком? {'Да' if ai_response.needs_human_review else 'Нет'}\n"
                    f"📋 Валидированные использованные источники:\n"
                )
                
                if ai_response.citations:
                    for doc in ai_response.citations:
                        log_message += f"  - Файл: {doc.source}, Страница: {doc.page}\n"
                else:
                    log_message += "  - [Источники отсутствуют (Честный ответ ИИ)]\n"
                    
                # Отправляем сформированный блок в логгер. Он напечатает текст 
                # в консоль VSCode и допишет его в файл storage/logs/app.log
                logger.info(log_message)
            else:
                logger.error(f"⚠️ Ошибка: Граф завершился успешно, но объект ответа 'final_answer' пуст.")

if __name__ == '__main__':
    # Запускаем асинхронное ядро на вашем ноутбуке
    asyncio.run(main())
