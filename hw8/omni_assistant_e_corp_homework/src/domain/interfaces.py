from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from src.domain.monads import Result

class BaseTelemetryProvider(ABC):
    """
    Базовый доменный интерфейс для всех ИИ-компонентов, 
    обязывающий их предоставлять LLMOps-метрики для профайлера.
    """
    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Возвращает словарь с накопленными метриками производительности (Latency, VRAM, Calls)."""
        pass

class BaseASREngine(BaseTelemetryProvider, ABC):
    """Абстрактный интерфейс для ушей ассистента (Audio-to-Text)."""
    @abstractmethod
    def transcribe(self, audio_path: Path, **kwargs: Any) -> Result[str]:
        """
        Преобразует входящую звуковую волну в текстовую строку.
        Поддерживает дополнительные именованные параметры (language, beam_size, fp16).
        """
        pass

class BaseVLMEngine(BaseTelemetryProvider, ABC):
    """Абстрактный интерфейс для глаз ассистента (Vision-Language)."""
    @abstractmethod
    def generate(self, image_path: Path, prompt_text: str, **kwargs: Any) -> Result[str]:
        """
        Выполняет мультимодальный анализ изображения совместно с текстовым промптом.
        Поддерживает дополнительные именованные параметры генерации.
        """
        pass

class BaseTTSEngine(BaseTelemetryProvider, ABC):
    """Абстрактный интерфейс для голоса ассистента (Text-to-Audio)."""
    @abstractmethod
    def synthesize(self, text: str, output_dir: Path, **kwargs: Any) -> Result[Path]:
        """
        Синтезирует речь на основе текста и запекает бинарный .wav файл на диск.
        Поддерживает дополнительные параметры (max_chars, speaker, sample_rate).
        """
        pass

class BaseOmniOrchestrator(ABC):
    """
    Интерфейс Сердца ассистента, связывающий все фазы в единый Pipeline.
    Обеспечивает сквозной контроль трансляции монад: Звук -> Текст -> Звук.
    """
    @abstractmethod
    def execute(self, image_path: Path, audio_query_path: Path) -> Result[Dict[str, Any]]:
        """
        Запускает асинхронный/синхронный защищенный конвейер инференса.
        Возвращает структурированный отчёт с результатами работы и метриками задержки (Latency).
        """
        pass

