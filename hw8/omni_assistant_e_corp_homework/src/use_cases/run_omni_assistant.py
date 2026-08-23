import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from src.domain.interfaces import BaseOmniOrchestrator
from src.domain.monads import Result
from config.logger import logger
from config.settings import settings

# Импортируем наши готовые инфраструктурные модули
from src.infrastructure.asr.whisper_engine import WhisperASREngine
from src.infrastructure.vlm.qwen_engine import QwenVLMEngine
from src.infrastructure.tts.silero_engine import SileroTTSEngine

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    asr: float = 0.0
    vlm: float = 0.0
    tts: float = 0.0
    total_pipeline: float = 0.0

class RunOmniAssistantUseCase(BaseOmniOrchestrator):
    """
    Сквозной оркестратор мультимодального конвейера (Сердце проекта).
    Последовательно связывает ASR, VLM и TTS фазы через защищенные контекстные блоки,
    выполняет LLMOps-профайлинг задержек и контролирует время выполнения.
    """
    def __init__(
        self,
        asr_engine: Optional[WhisperASREngine] = None,
        vlm_engine: Optional[QwenVLMEngine] = None,
        tts_engine: Optional[SileroTTSEngine] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        # Паттерн Dependency Injection
        self.asr = asr_engine if asr_engine is not None else WhisperASREngine()
        self.vlm = vlm_engine if vlm_engine is not None else QwenVLMEngine()
        self.tts = tts_engine if tts_engine is not None else SileroTTSEngine()
        
        self.output_audio_dir: Path = settings.PROJECT_ROOT / "data/output_audio"
        self.latency_log_path: Path = settings.LATENCY_LOGS
        self.timeout_seconds = timeout_seconds
        
        self.output_audio_dir.mkdir(parents=True, exist_ok=True)
        self.latency_log_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _time_block(self, name: str, metrics: LatencyMetrics):
        """Элегантный контекстный менеджер для безопасного замера Latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            setattr(metrics, name, elapsed)

    def _write_latency_log(self, metrics: LatencyMetrics, query_text: str) -> None:
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            # ИСПРАВЛЕНО: Строка объединена, убран разрыв, добавлен символ \n в самый конец
            log_line = (
                f"[{timestamp}] TOTAL: {metrics.total_pipeline:.3f}s | "
                f"ASR: {metrics.asr:.3f}s | VLM: {metrics.vlm:.3f}s | TTS: {metrics.tts:.3f}s | "
                f"Query length: {len(query_text)} chars\n"
            )
            with open(self.latency_log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            logger.error(f"⚠️ [PROFILER ERROR] Не удалось записать лог задержек на диск: {e}")

    def execute(self, image_path: Path, audio_query_path: Path) -> Result[Dict[str, Any]]:
        # Валидация входа (Fail-Fast)
        if not image_path.exists():
            return Result.failure("Файл изображения не найден.", "OMNI_INPUT_IMAGE_NOT_FOUND")
        if not audio_query_path.exists():
            return Result.failure("Аудиофайл запроса не найден.", "OMNI_INPUT_AUDIO_NOT_FOUND")
            
        logger.info("=== 🔥 [OMNI ORCHESTRATOR] Получен новый мультимодальный запрос ===")
        pipeline_start_time = time.perf_counter()
        metrics = LatencyMetrics()

        # Навешиваем программный сторожевой таймер на уровне рантайма (Защита от зависаний)
        # В условиях синхронного кода проверяем лимиты перед тяжелыми фазами
        
        # --- ШАГ 1: ФАЗА ASR (Уши) ---
        with self._time_block("asr", metrics):
            asr_result: Result[str] = self.asr.transcribe(audio_path=audio_query_path)
            
        if asr_result.is_failure():
            return Result.failure(
                f"Конвейер прерван на Фазе 1 (ASR): {asr_result.error}",
                f"OMNI_PIPELINE_ASR_FAILED_{asr_result.error_code}",
            )
            
        audio_query_text: str = asr_result.value
        if not audio_query_text.strip():
            return Result.failure(
                "Голосовой запрос пользователя пуст или содержит нераспознаваемый шум.",
                "OMNI_EMPTY_ASR_INPUT",
            )

        # Контроль общего таймаута
        if (time.perf_counter() - pipeline_start_time) > self.timeout_seconds:
            return Result.failure("Превышен лимит времени выполнения на этапе ASR.", "OMNI_TIMEOUT_EXCEEDED")

        # --- ШАГ 2: ФАЗА VLM (Глаза и Рассуждения) ---
        with self._time_block("vlm", metrics):
            vlm_result: Result[str] = self.vlm.generate(image_path=image_path, prompt_text=audio_query_text)
            
        if vlm_result.is_failure():
            return Result.failure(
                f"Конвейер прерван на Фазе 2 (VLM): {vlm_result.error}",
                f"OMNI_PIPELINE_VLM_FAILED_{vlm_result.error_code}",
            )
            
        vlm_response_text: str = vlm_result.value

        if (time.perf_counter() - pipeline_start_time) > self.timeout_seconds:
            return Result.failure("Превышен лимит времени выполнения на этапе VLM.", "OMNI_TIMEOUT_EXCEEDED")

        # --- ШАГ 3: ФАЗА TTS (Голос) ---
        with self._time_block("tts", metrics):
            tts_result: Result[Path] = self.tts.synthesize(text=vlm_response_text, output_dir=self.output_audio_dir)
            
        if tts_result.is_failure():
            return Result.failure(
                f"Конвейер прерван на Фазе 3 (TTS): {tts_result.error}",
                f"OMNI_PIPELINE_TTS_FAILED_{tts_result.error_code}",
            )
            
        output_audio_path: Path = tts_result.value
        
        # Считаем сквозную задержку
        metrics.total_pipeline = time.perf_counter() - pipeline_start_time
        self._write_latency_log(metrics, audio_query_text)
        
        report = {
            "total_tested": 1,
            "audio_query_text": audio_query_text,
            "vlm_response_text": vlm_response_text,
            "output_audio_path": str(output_audio_path),
            "latency_report_seconds": {
                "asr": metrics.asr,
                "vlm": metrics.vlm,
                "tts": metrics.tts,
                "total_pipeline": metrics.total_pipeline,
            },
        }
        
        logger.info("=== 🏁 [OMNI ORCHESTRATOR SUCCESS] Сквозной цикл инференса завершен успешно ===")
        logger.info(
            f" ⏱️ Общее время обработки: {metrics.total_pipeline:.3f} сек "
            f"(ASR: {metrics.asr:.2f}s | VLM: {metrics.vlm:.2f}s | TTS: {metrics.tts:.2f}s)"
        )
        return Result.success(report)

# --- Нативная процедурная обёртка для Gradio app.py ---
def run_omni_assistant(image: str, audio_query: str) -> tuple[str, str]:
    """Процедурная функция верхнего уровня, вызываемая веб-интерфейсом Gradio."""
    if not image or not audio_query:
        return "Ошибка: Загрузите изображение и запишите аудио-вопрос.", ""
        
    orchestrator = RunOmniAssistantUseCase()
    result = orchestrator.execute(image_path=Path(image), audio_query_path=Path(audio_query))
    
    if result.is_failure():
        return f"❌ Сбой Omni-пайплайна [{result.error_code}]: {result.error}", ""
        
    report = result.value
    return report["vlm_response_text"], report["output_audio_path"]
