from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from src.domain.monads import Result

# Импортируем нашу Pydantic-схему для строгого JSON-контракта
# (Если вы создали её в interfaces.py, оставьте описание класса выше, 
# если в отдельном файле schemas.py — импортируйте оттуда)
from pydantic import BaseModel, Field


class IntentSchema(BaseModel):
    """Схема строгого ответа маршрутизатора"""
    intent: Literal["corporate", "general"] = Field(description="Категория запроса: строго corporate или general")

class Citation(BaseModel):
    source: str = Field(description="Название файла источника")
    page: str = Field(description="Номер страницы")

class CorporateAnswerSchema(BaseModel):
    text_answer: str = Field(description="Текст ответа сотруднику")
    citations: List[Citation] = Field(description="Список источников")
    confidence_score: float = Field(description="Уверенность от 0 до 1")
    needs_human_review: bool = Field(description="Флаг проверки человеком")


class VectorStoreInterface(ABC):
    """Абстрактный контракт для векторной базы данных и гибридного поиска."""
    
    @abstractmethod
    def save_chunks(self, chunks: List[Dict[str, Any]]) -> Result[str]:
        """Сохранить нарезку чанков из парсера в БД."""
        pass

    @abstractmethod
    def search_similar(self, query: str, k: int) -> Result[List[Dict[str, Any]]]:
        """Найти топ-K релевантных документов."""
        pass


class LLMInterface(ABC):
    """Абстрактный контракт для локального ИИ-ядра генерации ответа."""

    @abstractmethod
    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> Result[str]:
        """Синхронная текстовая генерация (базовый метод)."""
        pass

    @abstractmethod
    async def generate_response_async(self, prompt: str, context: List[Dict[str, Any]], custom_system: Optional[str] = None) -> Result[str]:
        """Асинхронная текстовая генерация (используется на обычных шагах графа)."""
        pass

    @abstractmethod
    async def generate_structured_response_async(self, prompt: str, context: List[Dict[str, Any]], custom_system: str) -> Result[CorporateAnswerSchema]:
        """Асинхронная генерация строгого JSON, валидированного через Pydantic.
        
        Именно этот контракт гарантирует, что на выходе из LLM-узла 
        мы получим предсказуемый объект с цитатами, а не сырой текст.
        """
        pass
