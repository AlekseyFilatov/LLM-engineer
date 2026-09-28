from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue # Импорт gRPC фильтров
from config.logger import logger
from src.domain.monads import Result
from src.domain.interfaces import IVectorDbService


class QdrantVectorSearchService(IVectorDbService):
    """Сервис высокоскоростного поиска Qdrant DB с поддержкой Payload-фильтрации"""
    def __init__(self, host: str, port: int, collection: str):
        self.host = host
        self.port = port
        self.collection = collection
        self.client = QdrantClient(host=host, port=port, prefer_grpc=True)

    def search_candidates(self, query_vector: List[float], category_filter: str = None) -> Result[str]:
        try:
            # Динамически собираем С++ gRPC фильтр метаданных, если категория определена ИИ
            query_filter = None
            if category_filter:
                logger.info(f"🧱 [gRPC Filter] Активирован жесткий пре-фильтр категории: '{category_filter}'")
                query_filter = Filter(
                    must=[
                        FieldCondition(key="category", match=MatchValue(value=category_filter))
                    ]
                )

            try:
                search_result = self.client.query_points(
                    collection_name=self.collection, 
                    query=query_vector, 
                    using="text_vector",
                    query_filter=query_filter, # Применили фильтр в С++ контур
                    limit=5
                ).points
            except Exception:
                search_result = self.client.search(
                    collection_name=self.collection, 
                    query_vector=query_vector, 
                    target_vector="text_vector",
                    query_filter=query_filter, # Падающий fallback для старых версий
                    limit=5
                )
            
            if not search_result:
                return Result.success("Нет товаров.")
                
            candidates = []
            for hit in search_result:
                p = hit.payload
                candidates.append(
                    f"{"{"}'name': '{p.get('name')}', 'price': {p.get('price')}, "
                    f"'tags': '{p.get('tags')}', 'image_file': '{p.get('image_file')}', 'qdrant_score': {hit.score:.4f}{"}"}"
                )
            
            return Result.success(" | ".join(candidates))
        except Exception as e:
            return Result.failure(f"Ошибка базы Qdrant: {str(e)}", "QDRANT_DB_ERROR")
