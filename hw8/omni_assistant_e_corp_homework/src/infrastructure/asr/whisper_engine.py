import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import whisper
import librosa
from src.domain.interfaces import BaseASREngine
from src.domain.monads import Result
from config.logger import logger
from config.settings import settings

class WhisperASREngine(BaseASREngine):
    """
    Высокоскоростной защищённый оффлайн-движок ASR на базе OpenAI Whisper.
    Оптимизирован для продакшена: минимизация I/O, контроль ресурсов,
    динамическое шумоподавление, чанкинг с дедупликацией нахлестов слов и телеметрия.
    """
    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = "ru",
        device: Optional[str] = None,
        max_duration_seconds: float = 600.0,
        latency_window_size: int = 1000,
    ) -> None:
        self.model_dir: Path = Path(settings.ASR_DIR)
        self.model_name = model_name
        self.language = language
        self.max_duration = max_duration_seconds
        self.latency_window_size = latency_window_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self._model = None
        self._initialized = False
        
        # Накопительные промышленные метрики
        self._total_calls: int = 0
        self._failed_calls: int = 0
        self._latency_history: List[float] = []
        
        self._check_dependencies()

    @staticmethod
    def _check_dependencies() -> None:
        """Оффлайн-детектор наличия внешних библиотек шумоподавления."""
        try:
            import noisereduce  # noqa: F401
            WhisperASREngine._has_noisereduce = True
        except ImportError:
            WhisperASREngine._has_noisereduce = False

    @property
    def model(self):
        """Ленивый синглтон-доступ к аллокации весов."""
        if self._model is None or not self._initialized:
            self._initialize_model()
        return self._model

    def _initialize_model(self) -> None:
        if self._initialized:
            return
        logger.info("=== [ASR] Инициализация оффлайн-модели Whisper %s ===", self.model_name)
        try:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Директория весов Whisper не найдена: {self.model_dir}")
            
            logger.debug("📥 Загрузка весов Whisper из локального хранилища проекта...")
            try:
                self._model = whisper.load_model(
                    name=self.model_name,
                    device=self.device,
                    download_root=str(self.model_dir),
                )
            except RuntimeError as e:
                # Мягкий аппаратный откат при нехватке памяти GPU ноутбука
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ [ASR INITIALIZATION] Недостаточно памяти GPU, пробуем CPU фоллбэк...")
                    self.device = "cpu"
                    self._model = whisper.load_model(
                        name=self.model_name,
                        device=self.device,
                        download_root=str(self.model_dir),
                    )
                else:
                    raise
                    
            self._initialized = True
            logger.info("✅ [ASR INITIALIZATION] Whisper %s успешно развернут на %s.", self.model_name, self.device)
        except Exception as e:
            logger.exception("❌ [ASR INITIALIZATION] Сбой аллокации памяти под веса Whisper: %s", e)
            raise RuntimeError(f"Не удалось инициализировать модель Whisper: {e}") from e

    @staticmethod
    def _validate_audio_length(y: np.ndarray, sr: int, max_duration: float) -> None:
        if y is None or len(y) == 0:
            raise ValueError("Аудиофайл пуст или повреждён.")
        duration = len(y) / sr
        if duration > max_duration:
            raise ValueError(f"Аудио слишком длинное: {duration:.1f} сек (лимит: {max_duration:.1f} сек).")

    def _denoise_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        if getattr(WhisperASREngine, "_has_noisereduce", False):
            import noisereduce as nr
            return nr.reduce_noise(y=y, sr=sr, prop_decrease=0.85)
            
        # Защитный фоллбэк: классическое спектральное вычитание
        noise_duration_sec = 0.5
        noise_samples = int(sr * noise_duration_sec)
        if len(y) <= noise_samples:
            return y
            
        noise_sample = y[:noise_samples]
        stft_y = librosa.stft(y, n_fft=1024, hop_length=256)
        stft_n = librosa.stft(noise_sample, n_fft=1024, hop_length=256)
        
        mean_n = np.mean(np.abs(stft_n), axis=1, keepdims=True)
        subtracted = np.maximum(np.abs(stft_y) - mean_n, 0)
        cleaned_stft = subtracted * np.exp(1j * np.angle(stft_y))
        return librosa.istft(cleaned_stft, hop_length=256)

    @staticmethod
    def _normalize_audio(y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak == 0:
            return y
        return (y / peak) * 0.9

    def _preprocess_audio_in_memory(self, input_path: Path) -> np.ndarray:
        y, sr = librosa.load(str(input_path), sr=settings.TARGET_SAMPLE_RATE)
        self._validate_audio_length(y, sr, self.max_duration)
        y_denoised = self._denoise_audio(y, sr)
        return self._normalize_audio(y_denoised)

    def _transcribe_chunks(self, y: np.ndarray, sr: int, options: dict) -> str:
        chunk_len = int(settings.CHUNK_SECONDS * sr)
        overlap = int(settings.OVERLAP_SECONDS * sr)
        min_chunk = int(settings.MIN_CHUNK_SECONDS * sr)
        
        text_chunks: List[str] = []
        start = 0
        
        while start < len(y):
            end = min(start + chunk_len, len(y))
            chunk = y[start:end]
            if len(chunk) < min_chunk:
                break
                
            res = self.model.transcribe(chunk, **options)
            chunk_text = res.get("text", "").strip()
            if chunk_text:
                text_chunks.append(chunk_text)
                
            start += chunk_len - overlap
            
        if not text_chunks:
            return ""
        if len(text_chunks) == 1:
            return text_chunks[0]
            
        # Продвинутый алгоритм склейки нахлестов слов
        merged: List[str] = [text_chunks[0]]
        for i in range(1, len(text_chunks)):
            prev = merged[-1]
            curr = text_chunks[i]
            overlap_words = min(5, len(prev.split()), len(curr.split()))
            if overlap_words > 0:
                prev_tail = " ".join(prev.split()[-overlap_words:])
                curr_head = " ".join(curr.split()[:overlap_words])
                if prev_tail == curr_head:
                    curr = " ".join(curr.split()[overlap_words:]).strip()
            if curr:
                merged.append(curr)
                
        return " ".join(merged)

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        fp16: Optional[bool] = None,
    ) -> Result[str]:
        self._total_calls += 1
        if not audio_path.exists():
            self._failed_calls += 1
            return Result.failure(f"Аудиофайл не найден: {audio_path}", "ASR_FILE_NOT_FOUND")
            
        lang = language if language is not None else self.language
        fp16 = fp16 if fp16 is not None else (self.device == "cuda")
        
        try:
            logger.info("🎙️ [ASR] Запуск обработки входящего аудио-потока...")
            t_start = time.perf_counter()
            
            y_processed = self._preprocess_audio_in_memory(audio_path)
            options = {"language": lang, "beam_size": beam_size, "fp16": fp16}
            
            duration_sec = len(y_processed) / settings.TARGET_SAMPLE_RATE
            if duration_sec > settings.CHUNK_SECONDS + settings.OVERLAP_SECONDS:
                logger.info(f"⏳ Аудио длинное ({duration_sec:.1f} сек). Активирован режим чанкинга.")
                text_result = self._transcribe_chunks(y_processed, settings.TARGET_SAMPLE_RATE, options)
            else:
                transcribe_result = self.model.transcribe(y_processed, **options)
                text_result = transcribe_result.get("text", "").strip()
                
            latency = time.perf_counter() - t_start
            self._latency_history.append(latency)
            
            # Контроль скользящего окна метрик для исключения утечек памяти
            if len(self._latency_history) > self.latency_window_size:
                self._latency_history = self._latency_history[-self.latency_window_size:]
                
            logger.info(f"✅ [ASR SUCCESS] Аудио успешно обработано за {latency:.3f} сек.")
            return Result.success(text_result)
            
        except ValueError as ve:
            self._failed_calls += 1
            logger.warning("⚠️ [ASR VALIDATION] Ошибка валидации: %s", ve)
            return Result.failure(str(ve), "ASR_INVALID_AUDIO")
        except Exception as e:
            self._failed_calls += 1
            logger.exception("❌ [ASR CRASH] Критический сбой фазы ASR: %s", e)
            return Result.failure(f"Крах транскрибации: {str(e)}", "ASR_RUNTIME_ERROR")

    def get_telemetry(self) -> Dict[str, Any]:
        avg_latency = float(np.mean(self._latency_history)) if self._latency_history else 0.0
        return {
            "model_metadata": {"name": self.model_name, "device": self.device, "initialized": self._initialized},
	        "metrics": {"total_calls": self._total_calls, "failed_calls": self._failed_calls, "avg_latency_seconds": round(avg_latency, 3)}
        }