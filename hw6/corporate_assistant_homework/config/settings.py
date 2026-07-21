import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Загружает переменные из файла .env в окружение ОС
load_dotenv()

class Settings:
    # Базовые пути и токены
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    print(f"Токен успешно загружен: {bool(HF_TOKEN)}")

    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./storage/chroma_db")
    
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    def __init__(self):
        # Автоматически находим путь к prompts.yaml относительно этого файла конфигурации
        config_dir = Path(__file__).parent
        prompts_path = config_dir / "prompts.yaml"
        
        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f)
        else:
            self.prompts = {}

    # Удобные свойства-хелперы для быстрого доступа к промптам
    @property
    def corporate_keywords(self) -> list:
        return self.prompts.get("agent_brain", {}).get("corporate_keywords", [])

    @property
    def corporate_prompt(self) -> str:
        return self.prompts.get("generation_layer", {}).get("corporate_system_prompt", "")

    @property
    def general_prompt(self) -> str:
        return self.prompts.get("generation_layer", {}).get("general_system_prompt", "")

# Создаем глобальный синглтон настроек
settings = Settings()
