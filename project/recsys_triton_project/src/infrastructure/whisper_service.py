import os
import requests
from config.logger import logger
from src.domain.monads import Result

class WhisperASRService:
    """Сервис распознавания речи (Изолированный сетевой инференс Faster-Whisper)"""
    def __init__(self, url: str):
        # Автоматическая валидация URL: если передан базовый адрес,
        # дописываем канонический эндпоинт для Faster-Whisper / OpenAI спецификации
        self.url = url.strip()
        if not self.url.endswith("/asr") and not self.url.endswith("/v1/audio/transcriptions"):
            # Проверяем стандартный фолбек для большинства образов faster-whisper
            self.url = f"{self.url.rstrip('/')}/v1/audio/transcriptions"
            
    def transcribe(self, audio_path: str) -> Result[str]:
        try:
            logger.info("🗣️ [ASR] Старт обработки аудио реплики через Whisper...")
            logger.info(f"📡 [ASR Target] Отправка запроса на эндпоинт: {self.url}")
            
            if not audio_path or not os.path.exists(audio_path):
                return Result.failure(f"Файл аудиозаписи не найден: {audio_path}", "AUDIO_FILE_NOT_FOUND")

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                # Передаем файл сразу под двумя популярными ключами ('file' и 'audio_file')
                # для 100% совместимости с любыми кастомными API-интерфейсами
                files = {
                    'file': (os.path.basename(audio_path), audio_bytes, 'audio/wav'),
                    'audio_file': (os.path.basename(audio_path), audio_bytes, 'audio/wav')
                }
                
                # Стандартные параметры для OpenAI-совместимого эндпоинта
                data = {
                    'model': 'tiny', 
                    'language': 'ru',
                    'response_format': 'json'
                }
                
                response = requests.post(
                    self.url, 
                    files=files, 
                    data=data, 
                    timeout=15
                )
            
            # Если OpenAI эндпоинт выдал 404/405, пробуем фолбек на классический эндпоинт /asr
            if response.status_code in [404, 405] and "v1/audio/transcriptions" in self.url:
                fallback_url = self.url.replace("/v1/audio/transcriptions", "/asr")
                logger.info(f"🔄 [ASR Fallback] Метод отклонен. Пробуем альтернативный эндпоинт: {fallback_url}")
                
                with open(audio_path, "rb") as f:
                    files_fallback = {'audio_file': (os.path.basename(audio_path), f.read(), 'audio/wav')}
                    response = requests.post(fallback_url, files=files_fallback, data={'language': 'ru'}, timeout=15)

            # Удаляем временный файл Gradio
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось удалить временный файл Gradio: {e}")
            
            if response.status_code == 200:
                # Разные API возвращают либо {'text': ...}, либо {'transcription': ...}
                result_json = response.json()
                text = result_json.get("text", result_json.get("transcription", "")).strip()
                return Result.success(text)
                
            return Result.failure(f"Сбой Whisper API: {response.status_code} - {response.text}", "WHISPER_SERVER_ERROR")
            
        except Exception as e:
            return Result.failure(f"Исключение в модуле Whisper: {str(e)}", "WHISPER_EXCEPTION")


