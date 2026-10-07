from abc import ABC, abstractmethod
from typing import List
from src.domain.monads import Result

class IVectorDbService(ABC):
    """Интерфейс gRPC-контура векторной базы данных"""
    @abstractmethod
    def search_candidates(self, query_vector: List[float], category_filter: str = None) -> Result[str]:
        pass

class ITritonClientService(ABC):
    """Интерфейс распределенного контура инференса нейросетей"""
    @abstractmethod
    def generate_clip_embedding(self, search_tags: str) -> Result[List[float]]:
        pass

    @abstractmethod
    def compute_reranker_score(self, user_query: str, candidate_text: str) -> float:
        pass
