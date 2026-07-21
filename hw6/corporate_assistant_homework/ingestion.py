import os
import sys
import asyncio
import chromadb

# Настройка путей
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import settings
from config.logger import logger
from src.infrastructure.vector_stores import CorporateHybridSearchStore
from src.use_cases.ingest_documents import IngestDocumentsUseCase

async def run_ingestion():
    logger.info("🚀 Запуск процесса импорта документов в базу знаний...")
    
    # 1. Подключаемся к ChromaDB
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

    # Было:
    # collection = chroma_client.get_or_create_collection(name="corporate_knowledge")

    # Стало (Минимальные изменения для ультра-точного поиска):
    collection = chroma_client.get_or_create_collection(
        name="corporate_knowledge",
        metadata={
            "hnsw:space": "cosine",             # Явно задаем косинусное сходство [1]
            "hnsw:construction_ef": 200,        # Поднимаем точность при создании (дефолт: 100) [1]
            "hnsw:search_ef": 100,              # Поднимаем точность при поиске (дефолт: 10) [1]
            "hnsw:M": 32                        # Увеличиваем плотность связей графа (дефолт: 16) [1]
        }
    )
    
    # 2. Инициализируем хранилище и Use Case
    vector_store = CorporateHybridSearchStore(chroma_collection=collection)
    ingest_use_case = IngestDocumentsUseCase(vector_store=vector_store)
    
    # 3. Выполняем сценарий для папки data/
    data_directory = "./data"
    result = ingest_use_case.execute(data_directory)
    
    if result.is_failure():
        logger.error(f"❌ Импорт завершился ошибкой [{result.error_code}]: {result.error}")
    else:
        logger.info(f"✅ {result.value}")

if __name__ == "__main__":
    asyncio.run(run_ingestion())
