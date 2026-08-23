import os
import time
import re
import hashlib
import shelve
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torchaudio
import numpy as np
from num2words import num2words
import pymorphy3
import soundfile as sf

from src.domain.interfaces import BaseTTSEngine
from src.domain.monads import Result
from config.logger import logger
from config.settings import settings

logger = logging.getLogger(__name__)

class SileroTTSEngine(BaseTTSEngine):
    """
    Высокоскоростной защищённый оффлайн-движок синтеза речи (TTS) на базе Silero v5 API.
    Оптимизирован для продакшена: потокобезопасная загрузка через torch.package,
    персистентный NoSQL-кэш shelve в storage, лингвистический нормализатор pymorphy3.
    """
    def __init__(self, latency_window_size: int = 1000) -> None:
        self.model_path: Path = Path(settings.TTS_FILE)
        self.speaker: str = settings.TTS_SPEAKER
        self.sample_rate: int = settings.TTS_SAMPLE_RATE
        self.cache_db_path: Path = Path(settings.TTS_CACHE_DB)
        self.device = torch.device('cpu')       
        
        self._model = None
        self._initialized = False
        self._init_lock = torch.multiprocessing.Lock()
        
        self._total_syntheses: int = 0
        # Защита от Memory Leak: храним строго последние N замеров
        self._latency_history = deque(maxlen=latency_window_size)
        self._morph = pymorphy3.MorphAnalyzer()

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            logger.info("=== [TTS] Инициализация оффлайн-модели Silero v5 через torch.package ===")
            try:
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Файл весов Silero v5 не найден: {self.model_path}")
                if torch.get_num_threads() < 4:
                    torch.set_num_threads(4)                 
                
                logger.debug("📥 Загрузка локального package-контейнера Silero силами PyTorch...")
                
                # Загрузка JIT-монолита v5 через PackageImporter
                importer = torch.package.PackageImporter(str(self.model_path))
                self._model = importer.load_pickle('tts_models', 'model')
                self._model.to(self.device)            
                
                self._initialized = True
                logger.info(f"✅ [TTS INITIALIZATION] Локальный Silero v5 успешно прогрет на {self.device}.")
            except Exception as e:
                # Автоочистка: если файл битый, удаляем его, чтобы при следующем старте он перекачался чисто
                if self.model_path.exists():
                    try:
                        logger.warning(f"⚠️ Файл {self.model_path} поврежден, удаляем битый чекпоинт...")
                        self.model_path.unlink()
                    except OSError:
                        pass
                logger.exception("❌ [TTS INITIALIZATION] Сбой импорта package-весов Silero v5: %s", e)
                raise RuntimeError(f"Ошибка инициализации Silero v5: {e}") from e

    def _normalize_text_linguistic(self, text: str) -> str:
        """Настоящий лингвистический нормализатор числительных с согласованием падежей."""
        # ЭТАП 1: Санитария ДО токенизации (чтобы знаки препинания не липли к сокращениям)
        t_clean = re.sub(r"[*#|\\\[\]()_`{}]", "", text)
        # Раскрываем явные точки у бизнес-сокращений для корректного сплита
        t_clean = t_clean.replace("руб.", "рублей").replace("коп.", "копеек")
        t_clean = t_clean.replace("сек.", "секунд").replace("мин.", "минут")
        
        tokens = t_clean.split()
        normalized_tokens = []
        
        for idx, token in enumerate(tokens):
            if token.isdigit():
                gram_case = "nomn"
                if idx + 1 < len(tokens):
                    next_word = tokens[idx + 1].strip(".,!?")
                    parsed_list = self._morph.parse(next_word)
                    if parsed_list:
                        gram_case = parsed_list[0].tag.case if parsed_list[0].tag.case else "nomn"              
                try:
                    word_num_str = num2words(int(token), lang='ru', to='cardinal')
                    inflected_words = []
                    
                    for w_idx, word in enumerate(word_num_str.split()):
                        # Фикс для слова "сто" в родительном/косвенных падежах тысячных числительных
                        if word == "сто" and gram_case in ["gent", "datv", "ablt", "loct"]:
                            inflected_words.append("ста")
                            continue
                            
                        parsed_word_list = self._morph.parse(word)
                        if parsed_word_list:
                            parsed_word = parsed_word_list[0]
                            inflected_subword = parsed_word.inflect({gram_case})
                            inflected_words.append(inflected_subword.word if inflected_subword else word)
                        else:
                            inflected_words.append(word)
                    token = " ".join(inflected_words)
                except Exception:
                    try:
                        token = num2words(int(token), lang='ru')
                    except Exception:
                        pass
            normalized_tokens.append(token)                  
        
        return " ".join(normalized_tokens).strip()

    def synthesize(self, text: str, output_dir: Path, max_chars: int = 500) -> Result[Path]:
        if not text or not text.strip():
            return Result.failure("Передан пустой текст для синтеза речи.", "TTS_EMPTY_TEXT")
        
        clean_text = self._normalize_text_linguistic(text)       
        if len(clean_text) > max_chars:
            return Result.failure(f"Текст слишком длинный ({len(clean_text)} симв). Лимит: {max_chars}.", "TTS_TEXT_TOO_LONG")                    
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)      
        
        text_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
        final_wav_path = output_dir / f"tts_{text_hash}.wav"    
        
        # Проверка локального NoSQL кэш-базы shelve
        try:
            with shelve.open(str(self.cache_db_path)) as db:
                if text_hash in db:
                    cached_file = Path(db[text_hash])
                    if cached_file.exists() and cached_file.stat().st_size > 0:
                        logger.info("🎯 [TTS CACHE HIT] Возврат аудио из NoSQL-базы shelve.")
                        self._total_syntheses += 1
                        return Result.success(cached_file)
        except Exception as cache_err:
            logger.warning(f"⚠️ [TTS] Ошибка чтения базы кэша shelve: {cache_err}")            
            
        if not self._initialized:
            self._lazy_init()                       
            
        try:
            logger.info(f"🗣️ [TTS v5] Синтез речевой волны для диктора '{self.speaker}'...")
            t_start = time.perf_counter()           
            
            with torch.no_grad():
                # Вызов модели v5, загруженной через torch.package, идет через метод apply_tts
                audio_tensor = self._model.apply_tts(
                    text=clean_text,
                    speaker=self.speaker,
                    sample_rate=self.sample_rate
                )                
                
            # if audio_tensor.ndim == 1:
            #    audio_tensor = audio_tensor.unsqueeze(0)
                
            # Безопасно переводим float32 в PCM 16-бит для идеальной совместимости с HTML5-плеерами Gradio
            # audio_tensor = audio_tensor.clamp(-1.0, 1.0).mul(32767).to(torch.int16)
            
            # Сохранение на диск силами torchaudio
            # torchaudio.save(str(final_wav_path), audio_tensor, self.sample_rate)
            # Переводим PyTorch тензор в плоский NumPy массив, с которым работает soundfile
            audio_numpy = audio_tensor.squeeze().cpu().numpy()
            
            # Сохраняем на Ext4 диск в честном 16-битном формате WAV
            sf.write(str(final_wav_path), audio_numpy, self.sample_rate, format='WAV', subtype='PCM_16')
            
            if not final_wav_path.exists() or final_wav_path.stat().st_size == 0:
                raise FileNotFoundError("torchaudio не смог записать wav-матрицу на диск.")
                
            try:
                with shelve.open(str(self.cache_db_path)) as db:
                    db[text_hash] = str(final_wav_path)
            except Exception as cache_err:
                logger.error(f"❌ Ошибка записи в базу shelve: {cache_err}")               
                
            latency = time.perf_counter() - t_start
            self._total_syntheses += 1
            self._latency_history.append(latency)            
            logger.info(f"✅ [TTS SUCCESS] Речевой файл запечен в базу за {latency:.3f} сек.")
            return Result.success(final_wav_path)                 
            
        except Exception as e:
            logger.exception("❌ [TTS CRASH] Непредвиденный сбой инференса Silero v5: %s", e)
            if final_wav_path.exists():
                try: 
                    final_wav_path.unlink()
                except OSError: 
                    pass
            return Result.failure(f"Крах TTS v5 синтеза: {str(e)}", "TTS_RUNTIME_ERROR")

    def get_telemetry(self) -> Dict[str, Any]:
        avg_latency = float(np.mean(self._latency_history)) if self._latency_history else 0.0
        return {
            "model_metadata": {"name": "Silero-TTS-v5-RU", "speaker": self.speaker, "sample_rate": self.sample_rate, "device": str(self.device)},
            "metrics": {"total_syntheses": self._total_syntheses, "avg_latency_seconds": round(avg_latency, 3), "latency_samples": len(self._latency_history)}
        }
