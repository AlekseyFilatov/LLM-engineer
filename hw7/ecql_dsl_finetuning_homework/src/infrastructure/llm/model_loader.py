import os
import logging
from typing import Tuple, Optional, List, Dict, Any, Final
from packaging import version

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from src.domain.monads import Result
from config.settings import settings

logger = logging.getLogger(__name__)


import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Final
import torch

# Импортируем утилиту парсинга версий из экосистемы packaging
from packaging.version import parse

# Импортируем компоненты Hugging Face и PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

# Импортируем типы вашего проекта и синглтон настроек
# from your_system import Result, settings

logger = logging.getLogger(__name__)

class LocalModelTrainingLoader:
    MIN_VERSIONS: Final[Dict[str, str]] = {
        "transformers": "4.38.0",
        "accelerate": "0.27.0",
        "bitsandbytes": "0.42.0",
    }

    def __init__(self) -> None:

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Вместо абстрактного тега Ollama 'qwen2.5-coder:7b'
        # вычисляем абсолютный путь к локальным оригинальным весам Hugging Face на диске ext4
        # Путь: /home/alexfil/LLM-Training/src/storage/hf_cache/Qwen2.5-Coder-7B-Instruct
        self.model_path: Final[Path] = (Path(settings.PROJECT_ROOT) / "src" / "storage" / "hf_cache" / "Qwen2.5-Coder-7B-Instruct").resolve()
        self.cache_dir: Final[str] = str(self.model_path.parent)
        self.hf_token: Final[Optional[str]] = getattr(settings, "HF_TOKEN", None)

        # === ДИНАМИЧЕСКИЙ ТРИГГЕР ОФФЛАЙНА ===
        # Читаем из .env. По умолчанию включаем True (безопасный оффлайн)
        env_offline = os.getenv("OFFLINE_MODE", "True").lower() in ("true", "1", "yes")
        
        # Если папки физически нет на диске, принудительно отключаем оффлайн для скачивания
        if not self.model_path.exists() or not (self.model_path / "config.json").exists():
            self.is_offline = False
            self.model_identifier = "Qwen/Qwen2.5-Coder-7B-Instruct"
        else:
            self.is_offline = env_offline
            self.model_identifier = str(self.model_path) if self.is_offline else "Qwen/Qwen2.5-Coder-7B-Instruct"


    @staticmethod
    def _check_package_versions() -> Result[bool]:
        required_packages: Final[List[str]] = ["transformers", "accelerate", "bitsandbytes"]
        for pkg_name in required_packages:
            try:
                pkg = __import__(pkg_name)
                # ИСПРАВЛЕНО: Явный вызов импортированной функции parse
                current_ver = parse(pkg.__version__)
                min_ver = parse(LocalModelTrainingLoader.MIN_VERSIONS[pkg_name])
                if current_ver < min_ver:
                    return Result.failure(
                        f"Пакет {pkg_name} версии {current_ver} ниже минимально требуемой {min_ver}",
                        "ENV_VERSION_ERROR",
                    )
            except ImportError as e:
                return Result.failure(
                    f"Отсутствует обязательный ML-пакет для Fine-tuning: {pkg_name}. Сбой: {str(e)}", 
                    "ENV_IMPORT_ERROR"
                )
        return Result.success(True)

    @staticmethod
    def _ensure_cuda_available() -> Result[bool]:
        if not torch.cuda.is_available():
            return Result.failure(
                "Драйвер CUDA не зафиксирован. 4-битное квантование и обучение требуют GPU.", 
                "CUDA_HARDWARE_UNAVAILABLE"
            )
        return Result.success(True)

    def _prepare_tokenizer(self) -> Result[AutoTokenizer]:
        try:
            # Используем динамические переменные identifier и is_offline
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=self.is_offline  # Автоматический переключатель сети
            )
        except Exception as e:
            return Result.failure(f"Не удалось загрузить токенизатор: {str(e)}", "TOKENIZER_LOAD_FAILED")
            
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                
        tokenizer.padding_side = "left"
        special_tokens: List[str] = getattr(settings, "dsl_special_tokens", [])
        if special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        return Result.success(tokenizer)

    def _load_base_model(self, tokenizer: AutoTokenizer) -> Result[AutoModelForCausalLM]:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        try:
            # Используем динамические переменные identifier и is_offline
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_identifier,
                quantization_config=bnb_config,
                device_map="auto",
                #trust_remote_code=True,
                trust_remote_code=False,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=self.is_offline  # Автоматический переключатель сети
            )
        except Exception as e:
            return Result.failure(f"Ошибка загрузки базовых весов модели: {str(e)}", "MODEL_LOAD_FAILED")
            
        if len(tokenizer) != base_model.config.vocab_size:
            logger.info("📐 Расширяем матрицу эмбеддингов под новый размер токенизатора: %d", len(tokenizer))
            base_model.resize_token_embeddings(len(tokenizer))
            
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
        return Result.success(base_model)

    def _apply_lora(self, base_model: AutoModelForCausalLM) -> Result[PeftModel]:
        lora_data: Dict[str, Any] = getattr(settings, "lora_config_data", {})
        required_keys: Final[List[str]] = ["r", "alpha", "target_modules", "dropout"]
        missing_keys = [k for k in required_keys if k not in lora_data]
        if missing_keys:
            return Result.failure(
                f"В конфигурационном YAML prompts.yaml отсутствуют обязательные ключи LoRA: {missing_keys}", 
                "LORA_CONFIG_INVALID"
            )
            
        lora_config = LoraConfig(
            r=lora_data["r"],
            lora_alpha=lora_data["alpha"],
            target_modules=lora_data["target_modules"],
            lora_dropout=lora_data.get("dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
        )
        try:
            # Накладываем адаптер LoRA на слои внимания модели
            model = get_peft_model(base_model, lora_config)
            # ИСПРАВЛЕНО: Дублирующий вызов gradient_checkpointing_enable() удален для защиты от RuntimeError
        except Exception as e:
            return Result.failure(f"Ошибка наложения LoRA адаптера на слои внимания/MLP: {str(e)}", "LORA_APPLY_FAILED")
        return Result.success(model)

    @staticmethod
    def _smoke_test_model(model: PeftModel, tokenizer: AutoTokenizer) -> Result[bool]:
        device = model.device
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        test_input = torch.tensor([[pad_id]], device=device)
        with torch.no_grad():
            try:
                outputs = model(test_input)
                logits = outputs.logits               
                if torch.isnan(logits).any():
                    return Result.failure("Smoke-тест провален: обнаружены аномалии NaN в логитах модели. Квантование повреждено.", "SMOKE_TEST_FAILED")
                if not torch.isfinite(logits).all():
                    return Result.failure("Smoke-тест провален: обнаружены бесконечные значения Inf/-Inf в логитах.", "SMOKE_TEST_FAILED")
            except Exception as e:
                return Result.failure(f"Аппаратный Smoke-тест завершился системной ошибкой ядра PyTorch: {str(e)}", "SMOKE_TEST_FAILED")            
        return Result.success(True)

    def load_and_prepare_for_training(self) -> Result[Tuple[PeftModel, AutoTokenizer]]:
        env_check = self._check_package_versions()
        if env_check.is_failure(): return Result.failure(env_check.error, env_check.error_code)
        
        cuda_check = self._ensure_cuda_available()
        if cuda_check.is_failure(): return Result.failure(cuda_check.error, cuda_check.error_code)
        
        tokenizer_res = self._prepare_tokenizer()
        if tokenizer_res.is_failure(): return Result.failure(tokenizer_res.error, tokenizer_res.error_code)
        tokenizer = tokenizer_res.value
        
        model_res = self._load_base_model(tokenizer)
        if model_res.is_failure(): return Result.failure(model_res.error, model_res.error_code)
        base_model = model_res.value
        
        lora_res = self._apply_lora(base_model)
        if lora_res.is_failure(): return Result.failure(lora_res.error, lora_res.error_code)
        model = lora_res.value
        
        smoke_res = self._smoke_test_model(model, tokenizer)
        if smoke_res.is_failure(): return Result.failure(smoke_res.error, smoke_res.error_code)
        
        return Result.success((model, tokenizer))
