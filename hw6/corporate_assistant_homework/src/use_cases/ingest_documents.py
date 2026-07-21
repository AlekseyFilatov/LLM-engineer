from src.domain.interfaces import VectorStoreInterface
from src.infrastructure.parser.pdf_parser import AdvancedDocumentParser
from src.domain.monads import Result
import os

class IngestDocumentsUseCase:
    def __init__(self, vector_store: VectorStoreInterface):
        self.vector_store = vector_store
        self.parser = AdvancedDocumentParser()

    def execute(self, data_dir: str) -> Result[str]:
        """Безопасный сценарий потокового парсинга и загрузки в БД"""
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            return Result.failure(f"Папка с документами '{data_dir}' пуста или не существует.", "EMPTY_DATA_DIR")
        
        try:
            chunks_to_load = []
            
            # Поточно собираем чанки через наш генератор
            for chunk in self.parser.parse_directory_generator(data_dir):
                chunks_to_load.append(chunk)
                
            if not chunks_to_load:
                return Result.failure("Не найдено валидных PDF документов для парсинга.", "NO_VALID_CHUNKS")

            # Передаем чанки в интерфейс нашей векторной базы данных
            # Метод save_chunks мы вызовем у нашей будущей реализации ChromaDB/BM25
            self.vector_store.save_chunks(chunks_to_load)
            
            return Result.success(f"Успешно обработано и добавлено чанков: {len(chunks_to_load)}")
            
        except Exception as e:
            return Result.failure(
                error_message=f"Критический сбой на этапе загрузки документов: {str(e)}",
                code="INGESTION_PIPELINE_ERROR"
            )
