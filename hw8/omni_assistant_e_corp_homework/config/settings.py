import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if (CURRENT_DIR.name == 'config' or CURRENT_DIR.name == 'src') else CURRENT_DIR
load_dotenv(PROJECT_ROOT / '.env')

class OmniSettings:
    PROJECT_ROOT: Path = PROJECT_ROOT
    
    # Репозитории (для скрипта скачивания)
    VLM_REPO: str = os.getenv('OMNI_VLM_REPO', 'Qwen/Qwen2-VL-2B-Instruct')
    ASR_REPO: str = os.getenv('OMNI_ASR_REPO', 'openai/whisper-base')
    TTS_URL: str = os.getenv('OMNI_TTS_URL', 'https://models.silero.ai/models/tts/ru/v5_ru.pt')
    
    # Локальные абсолютные пути хранения (с гарантией конвертации в Path)
    HF_HOME: Path = PROJECT_ROOT / os.getenv('STORAGE_HF_HOME', 'src/storage/hf_home_cache')
    VLM_DIR: Path = PROJECT_ROOT / os.getenv('STORAGE_VLM_DIR', 'src/storage/hf_cache/Qwen2-VL-2B-Instruct')
    ASR_DIR: Path = PROJECT_ROOT / os.getenv('STORAGE_ASR_DIR', 'src/storage/whisper_cache')
    TTS_DIR: Path = PROJECT_ROOT / os.getenv('STORAGE_TTS_DIR', 'src/storage/silero_cache')
    TTS_FILE: Path = PROJECT_ROOT / os.getenv('STORAGE_TTS_FILE', 'src/storage/silero_cache/v5_ru.pt')

    TTS_CACHE_DB: Path = TTS_DIR / 'tts_cache.db'

    # Параметры инференса и профилирования
    MAX_NEW_TOKENS: int = int(os.getenv('OMNI_MAX_NEW_TOKENS', 128))
    LATENCY_LOGS: Path = PROJECT_ROOT / os.getenv('OMNI_LATENCY_LOGS', 'data/latency_metrics.log')

    # Аудио-константы для WhisperEngine
    TARGET_SAMPLE_RATE: int = int(os.getenv('OMNI_TARGET_SAMPLE_RATE', 16000))
    CHUNK_SECONDS: int = int(os.getenv('OMNI_CHUNK_SECONDS', 30))
    OVERLAP_SECONDS: int = int(os.getenv('OMNI_OVERLAP_SECONDS', 1))
    MIN_CHUNK_SECONDS: float = float(os.getenv('OMNI_MIN_CHUNK_SECONDS', 0.5))

    # Переменные для QwenVLMEngine (Лимиты рантайма)
    VLM_MAX_RESPONSE_TOKENS: int = int(os.getenv('VLM_MAX_RESPONSE_TOKENS', 128))
    VLM_MAX_LATENCY_HISTORY: int = int(os.getenv('VLM_MAX_LATENCY_HISTORY', 1000))
    
    # Промпт по умолчанию (Заменится данными из YAML при инициализации)
    VLM_SYSTEM_PROMPT: str = os.getenv('VLM_SYSTEM_PROMPT', 'Ответь лаконично по изображению.')

    # Настройки для Silero TTS
    TTS_SPEAKER: str = os.getenv('OMNI_TTS_SPEAKER', 'kseniya')
    TTS_SAMPLE_RATE: int = int(os.getenv('OMNI_TTS_SAMPLE_RATE', 24000))

    def __init__(self):
        # 1. Автоматически прописываем HF_HOME в рантайм ОС для изоляции кэша
        os.environ['HF_HOME'] = str(self.HF_HOME)
        
        # 2. Интеграция внешнего YAML промпт-контейнера
        self._load_external_prompts()

    def _load_external_prompts(self) -> None:
        """Безопасный оффлайн-парсер внешнего реестра промптов."""
        # Ищем prompts.yaml как внутри папки config/, так и в корне проекта
        prompts_path = self.PROJECT_ROOT / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = self.PROJECT_ROOT / "prompts.yaml"

        if prompts_path.exists():
            try:
                with open(prompts_path, "r", encoding="utf-8") as f:
                    prompt_data = yaml.safe_load(f)
                
                # Извлекаем системный промпт из структуры YAML
                if prompt_data and "vlm" in prompt_data and "system_prompt" in prompt_data["vlm"]:
                    raw_prompt = prompt_data["vlm"]["system_prompt"]
                    # Очищаем строку от лишних переносов строк, сохраняя плоский текст
                    self.VLM_SYSTEM_PROMPT = " ".join(raw_prompt.split()).strip()
                    logger.info("📝 [SETTINGS] Системный промпт VLM успешно импортирован из config/prompts.yaml")
            except Exception as e:
                logger.warning(f"⚠️ [SETTINGS ERROR] Не удалось прочитать prompts.yaml ({e}). Используется фоллбэк.")
        else:
            logger.info("ℹ️ [SETTINGS] Файл config/prompts.yaml не найден. Используется встроенный системный промпт.")

settings = OmniSettings()
