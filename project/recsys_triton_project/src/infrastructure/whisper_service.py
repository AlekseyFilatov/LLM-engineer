import os
import requests
import numpy as np
from scipy.io import wavfile
from config.logger import logger
from src.domain.monads import Result

class WhisperASRService:
    """Сервис распознавания речи (Изолированный сетевой инференс Faster-Whisper)"""
    def __init__(self, url: str):
        self.url = url

    def transcribe(self, audio_path: str) -> Result[str]:
        try:
            logger.info("🗣️ [ASR] Старт обработки аудио реплики через Whisper...")
            
            # ИСПРАВЛЕНО: Всеядный FFmpeg-конвертер. Переводит MP3/OGG в чистый 16kHz WAV
            converted_wav = "converted_input.wav"
            # Глушим лишние логи ffmpeg через -y -loglevel quiet
            os.system(f"ffmpeg -y -i '{audio_path}' -ar 16000 -ac 1 -c:a pcm_s16le -loglevel quiet '{converted_wav}'")
            
            if not os.path.exists(converted_wav) or os.path.getsize(converted_wav) == 0:
                return Result.failure("FFmpeg не смог перекодировать аудиофайл.", "FFMPEG_CONVERSION_ERROR")

            # Теперь scipy гарантированно прочитает RIFF-заголовок без исключений!
            sample_rate, audio_np = wavfile.read(converted_wav)
            audio_np = audio_np.astype(np.float32) / 32768.0
            if len(audio_np.shape) > 1: 
                audio_np = audio_np[:, 0]
            
            tmp_file = "tmp_voice.wav"
            wavfile.write(tmp_file, sample_rate, (audio_np * 32768.0).astype(np.int16))
            
            with open(tmp_file, "rb") as f:
                files = {'file': ('audio.wav', f.read(), 'audio/wav')}
                response = requests.post(
                    self.url, files=files, 
                    data={'model': 'tiny', 'language': 'ru'}, 
                    timeout=10
                )
            
            # Зачищаем за собой временные файлы конвертации
            for f_path in [converted_wav, tmp_file]:
                if os.path.exists(f_path): 
                    os.remove(f_path)
            
            if response.status_code == 200:
                text = response.json().get("text", "").strip()
                return Result.success(text)
            return Result.failure(f"Сбой Whisper API: {response.text}", "WHISPER_SERVER_ERROR")
            
        except Exception as e:
            return Result.failure(f"Исключение в модуле Whisper: {str(e)}", "WHISPER_EXCEPTION")
