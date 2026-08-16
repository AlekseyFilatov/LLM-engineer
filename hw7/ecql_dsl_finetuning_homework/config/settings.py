import os
import yaml
from pathlib import Path

import os
import yaml
from pathlib import Path

class Settings:
    # 1. Вычисляем корень проекта динамически (каталог, где развернуто приложение)
    # Это страхует от любых проблем с путями при запуске из разных папок
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "config" else Path(__file__).resolve().parent

    # 2. Обновленные пути с учетом структуры /src/storage/ под нативный Linux ext4
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1")
    #MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "/home/alexfil/LLM-Training/src/storage/hf_cache/Qwen2.5-Coder-7B-Instruct")
    TOTAL_TARGET_SAMPLES: int = int(os.getenv("TOTAL_TARGET_SAMPLES", 200))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.75))
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", 50))
    
    # Исправлено: пути теперь смотрят строго внутрь /src/storage/
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "./src/storage/embedding_model")
    DATA_SCHEMA_PATH: str = os.getenv("DATA_SCHEMA_PATH", "./config/data_schema.yaml")

    def __init__(self):
        # Строим железно работающий абсолютный путь к prompts.yaml от корня проекта
        config_dir = self.PROJECT_ROOT / "config"
        prompts_path = config_dir / "prompts.yaml"
        
        if prompts_path.exists():
            with open(prompts_path, "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f) or {}
        else:
            self.prompts = {}

    @property
    def ecql_generation_prompt(self) -> str:
        return self.prompts.get("agent_brain", {}).get("ecql_generation", "")

    @property
    def lora_config_data(self) -> dict:
        return self.prompts.get("finetuning", {}).get("lora", {
            "r": 16, "alpha": 32, "dropout": 0.05, 
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        })
    
    @property
    def training_hyperparameters(self) -> dict:
        return self.prompts.get("finetuning", {}).get("hyperparameters", {})
    
    @property
    def data_catalog_string(self) -> str:
        """Возвращает схему данных в виде чистого форматированного текста без багов путей."""
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Разрешаем абсолютный путь от корня проекта
        schema_path = (self.PROJECT_ROOT / self.DATA_SCHEMA_PATH).resolve()
        
        if not schema_path.exists():
            return "### КОРПОРАТИВНЫЙ КАТАЛОГ ДАННЫХ E-CORP:\nСхема данных пуста. Проверить путь: " + str(schema_path)
            
        with open(schema_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        lines = ["### КОРПОРАТИВНЫЙ КАТАЛОГ ДАННЫХ E-CORP:"]
        for entity, info in data.get("entities", {}).items():
            lines.append(f"Сущность: [{entity}] ({info.get('description', '')})")
            lines.append("Доступные поля:")
            for field, desc in info.get("fields", {}).items():
                lines.append(f"  - {field} : {desc}")
        return "\n".join(lines)
    
    @property
    def dsl_special_tokens(self) -> list:
        return self.prompts.get("finetuning", {}).get("special_tokens", ["&&", "||", "@", "[", "]"])

    @property
    def dup_detector_page_size(self) -> int:
        return int(os.getenv("DUP_DETECTOR_CACHE_PAGE_SIZE", 5000))

    @property
    def dup_detector_n_results(self) -> int:
        return int(os.getenv("DUP_DETECTOR_ANN_N_RESULTS", 5))

    @property
    def dup_detector_default_threshold(self) -> float:
        return float(os.getenv("DUP_DETECTOR_DEFAULT_THRESHOLD", 0.85))

# Центральный синглтон объекта
settings = Settings()

