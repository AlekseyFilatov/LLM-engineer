import sys
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from src.domain.monads import Result
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Set, Optional, Literal
from pathlib import Path


class ECQLPair(BaseModel):
    """Схема одной валидной пары данных."""
    input_text: str = Field(alias="input", description="Запрос на естественном русском языке")
    output_text: str = Field(alias="output", description="Точный эквивалент запроса на вымышленном корпоративном языке ECQL")

class BatchGenerationResponse(BaseModel):
    """Схема батча, возвращаемого моделью-учителем за одну итерацию."""
    pairs: List[ECQLPair] = Field(description="Список сгенерированных пар запросов")

# --- ДОБАВЛЕНО: Схема строгого ответа ИИ-маршрутизатора ---
class IntentSchema(BaseModel):
    """Схема валидации семантической категории запроса."""
    intent: Literal["corporate", "general"] = Field(
        description="Категория запроса: строго corporate (регламенты, правила, DSL) или general (беседа, приветствие)"
    )

# --- ИСПРАВЛЕНО: Очищенное и масштабируемое состояние графа ---
class PipelineState(TypedDict):
    """Состояние итеративного графа LangGraph."""
    config: Dict[str, Any]                   # Загруженный YAML-конфиг
    task_pool: List[Dict[str, Any]]           # Финальный пул уникальных пар {"input": ..., "output": ...}
    current_candidates: List[Dict[str, Any]]     # Временный буфер кандидатов текущей итерации
    loop_count: int                          # Текущий счетчик итераций
    error_rail: Result                       # Монадическая рельса безопасности

class IntentSchema(BaseModel):
    """Схема валидации семантической категории запроса."""
    intent: Literal["corporate", "general"] = Field(
        description="Категория запроса: строго corporate (регламенты, правила, DSL) или general (беседа, приветствие)"
    )

class CorporateAnswerSchema(BaseModel):
    """Схема строгого структурированного ответа ИИ-маршрутизатора для корпоративного контура."""
    reasoning: str = Field(description="Пошаговая логика и семантический разбор запроса пользователя")
    is_dsl_applicable: bool = Field(description="Флаг необходимости перевода запроса на язык ECQL")
    confidence_score: float = Field(description="Уверенность классификатора от 0.0 до 1.0")


# --- 2. ДОБАВЛЕНО: Абстрактный интерфейс для LLM-клиента (Port/Interface) ---

class LLMInterface(ABC):
    """Абстрактный интерфейс OpenAI-совместимого локального клиента Ollama/vLLM."""

    @abstractmethod
    async def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        """Синхронный/Асинхронный инференс модели в свободном текстовом формате."""
        pass

    @abstractmethod
    async def generate_structured(self, prompt: str, response_model: Any, temperature: float = 0.0) -> Any:
        """Инференс модели со строгой Compiler-in-the-Loop валидацией выходной Pydantic-схемы.
        
        Используется для Intent Routing (IntentSchema) и разбора схемы данных.
        """
        pass

class IVectorDuplicateDetector(ABC):
    """Абстрактный интерфейс детектора дубликатов корпоративного уровня."""

    @abstractmethod
    def refresh_cache(self) -> None:
        """Инвалидация и полная перезагрузка ОЗУ-кэша хэшей кода из постоянного хранилища."""
        pass

    @abstractmethod
    def check_duplicates_and_update(
        self, 
        new_pairs: List[Dict[str, str]], 
        threshold: Optional[float] = None
    ) -> Result[List[bool]]:
        """Двухфакторная проверка пачки кандидатов (In-Batch + ANN в БД) с записью уникальных.
        
        Возвращает булеву маску дубликатов, где True — дубликат (отклонить), False — уникален.
        """
        pass

class IECQLValidator(ABC):
    """Абстрактный интерфейс детерминированного компилятора-валидатора ECQL."""

    @classmethod
    @abstractmethod
    def validate_syntax(cls, query: str) -> Result[str]:
        """Выполняет лексический, синтаксический анализ и семантическую проверку типов данных (Type Checking)."""
        pass


class IModelTrainingLoader(ABC):
    """Абстрактный интерфейс локального загрузчика и изолятора базовой LLM."""

    @abstractmethod
    def load_and_prepare_for_training(self) -> Result[Tuple[Any, Any]]:
        """Проверяет окружение, драйвер CUDA, загружает квантованную базовую модель и накладывает LoRA.
        
        Возвращает кортеж (PeftModel, AutoTokenizer).
        """
        pass


class IDatasetHandler(ABC):
    """Абстрактный интерфейс безопасной пакетной выгрузки и верификации датасетов."""

    @abstractmethod
    def format_for_sft(self, train_dataset_path: str) -> Result[Any]:
        """Считывает данные, верифицирует контракт полей и возвращает валидный DatasetDict."""
        pass


class ITrainerFactory(ABC):
    """Абстрактный интерфейс фабрики конфигурирования аппаратно-оптимизированного SFTTrainer."""

    @abstractmethod
    def create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        config: Any
    ) -> Result[Any]:
        """Собирает TrainingArguments, настраивает DataCollator с маскированием и возвращает инстанс SFTTrainer."""
        pass