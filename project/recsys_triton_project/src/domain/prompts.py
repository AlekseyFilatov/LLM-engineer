import yaml
from pathlib import Path
from typing import Dict, Any
from config.logger import logger

class PromptLoader:
    """Декларативный загрузчик и конфигуратор системных промптов из YAML"""
    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        self.config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"📦 [CONFIG] Промпты из {self.yaml_path.name} успешно загружены.")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки prompts.yaml: {e}")
            self.config = {}

    def build_prompt(self, stage: str, **kwargs) -> str:
        stage_cfg = self.config.get(stage, {})
        template = stage_cfg.get("template", "{user_query}")
        render_args = {
            "system_prompt": stage_cfg.get("system_prompt", ""),
            "few_shot_example": stage_cfg.get("few_shot_example", ""),
            **kwargs
        }
        return template.format(**render_args)
