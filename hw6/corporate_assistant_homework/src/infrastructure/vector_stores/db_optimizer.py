import os
import sqlite3
import chromadb
from src.domain.monads import Result
from config.settings import settings

class ChromaDBOptimizer:
    """Сервис обслуживания и оптимизации локальной ChromaDB."""
    def __init__(self, db_path: str):
        self.db_path = db_path

    def optimize_and_validate(self) -> Result[str]:
        try:
            # 1. Оптимизация SQLite (таблицы метаданных и ID)
            # Находим файл базы данных ChromaDB
            sqlite_file = os.path.join(self.db_path, "chroma.sqlite3")
            
            if os.path.exists(sqlite_file):
                print(f"📦 [DB OPTIMIZER] Сжатие таблиц метаданных: {sqlite_file}")
                conn = sqlite3.connect(sqlite_file)
                cursor = conn.cursor()
                # Выполняем дефрагментацию дискового пространства
                cursor.execute("VACUUM;")
                # Оптимизируем внутренние поисковые индексы SQLite
                cursor.execute("ANALYZE;")
                conn.commit()
                conn.close()
            
            # 2. Прогрев кэша (Cache Warming) для векторного индекса HNSW
            print("🔥 [DB OPTIMIZER] Прогрев векторного кэша HNSW...")
            client = chromadb.PersistentClient(path=self.db_path)
            collection = client.get_or_create_collection(name="corporate_knowledge")
            
            # Делаем холостой легкий запрос, чтобы HNSW-индекс загрузился с диска D в ОЗУ
            _ = collection.query(query_texts=["холостой запрос для прогрева кэша"], n_results=1)
            
            return Result.success("ChromaDB успешно оптимизирована, индексы перестроены, кэш прогрет!")
            
        except Exception as e:
            return Result.failure(f"Сбой при оптимизации ChromaDB: {str(e)}", "DB_OPTIMIZATION_ERROR")
