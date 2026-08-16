import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Final

import torch
from pydantic import BaseModel, Field, ValidationError
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizerFast
from trl import SFTTrainer, SFTConfig 

from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizerFast, 
    TrainingArguments, 
    DataCollatorForLanguageModeling 
)

from src.domain.monads import Result
from config.settings import settings

logger = logging.getLogger(__name__)

from peft import PeftModel

class CustomCompletionOnlyCollator:
    """Высокопроизводительный маскировщик контекста на чистом PyTorch.
    Устойчив к изменениям в TRL 0.13+: безопасно выравнивает матрицы и маскирует промпт.
    """
    def __init__(self, response_template: str, tokenizer: Any):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        clean_examples = []
        for ex in examples:
            # Очищаем пример от текста И от сырых разнодлинных полей 'labels'
            clean_ex = {
                k: v for k, v in ex.items() 
                if k in ["input_ids", "attention_mask"] or isinstance(v, list)
            }
            # ГАРАНТИЯ ОФФЛАЙН-ПАДДИНГА: Насильно вырезаем 'labels', если TRL подсунул их раньше времени
            clean_ex.pop("labels", None)
            clean_examples.append(clean_ex)

        # Выравниваем input_ids и attention_mask нативного батча
        batch = self.tokenizer.pad(
            clean_examples,
            padding=True,
            return_tensors="pt"
        )
        
        # Создаем labels как идеальную, уже выровненную по длине копию input_ids
        labels = batch["input_ids"].clone()
        
        # Построчно маскируем контекст пользователя до ответа ассистента
        # Используем labels.shape[0] вместо labels.shape, чтобы получить размер батча (количество строк)
        for i in range(labels.shape[0]):
            input_ids_list = batch["input_ids"][i].tolist()
            
            idx = -1
            for j in range(len(input_ids_list) - len(self.response_token_ids) + 1):
                if input_ids_list[j:j+len(self.response_token_ids)] == self.response_token_ids:
                    idx = j + len(self.response_token_ids)
                    break
            
            # Маскируем всё, что идет ДО начала ответа ассистента маркёром -100
            if idx != -1:
                labels[i, :idx] = -100
            else:
                labels[i, :] = -100
                
        batch["labels"] = labels
        return batch

class TrainingHyperparameters(BaseModel):
    """Валидационная Pydantic-модель ML-гиперпараметров градиентного спуска."""
    num_train_epochs: int = Field(default=3, gt=0, le=20)
    per_device_train_batch_size: int = Field(default=2, gt=0, le=64)
    gradient_accumulation_steps: int = Field(default=4, gt=0, le=128)
    learning_rate: float = Field(default=2e-4, gt=0.0, lt=1e-1)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    
    # Улучшено: Косинусный шедулер для идеального схождения лосса
    lr_scheduler_type: str = Field(default="cosine")
    
    # Улучшено: Оптимизированный разогрев под маленький датасет
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    # Вместо: warmup_ratio=getattr(config, 'warmup_ratio', 0.03),
    # Напишем явное количество шагов прогрева (warmup_steps). 
    # Оптимально для вашего датасета — 3-5 шагов:
    # warmup_steps=5, 
    
    # ИСПРАВЛЕНО: Строгое имя 8-битного оптимизатора BitsAndBytes
    optim: str = Field(default="paged_adamw_8bit")
    
    logging_steps: int = Field(default=1, gt=0)
    
    # ИСПРАВЛЕНО: Путь синхронизирован с нативной структурой проекта
    output_dir: str = Field(default="./src/storage/lora_output")
    
    # Отличный выбор сида! В сообществе шутят, что сид 3407 дает +1% к точности. Оставляем.
    seed: int = Field(default=3407)
    
    enable_packing: bool = Field(default=False)
    max_seq_length: Optional[int] = Field(default=None)


logger = logging.getLogger(__name__)

class LocalTrainerFactory:
    
    # Официальный ChatML шаблон для моделей семейства Qwen 2.5
    QWEN_CHAT_TEMPLATE: Final[str] = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    @staticmethod
    def _get_response_template(tokenizer: PreTrainedTokenizerFast) -> str:
        """Определяет точный маркер начала генерации ассистента на основе имени модели."""
        model_name: str = (settings.MODEL_NAME or "").lower()
        if "mistral" in model_name or "llama" in model_name:
            return "[/INST]"
        elif "qwen" in model_name:
            return "<|im_start|>assistant\n"
        else:
            return "<|im_start|>assistant\n"

    @staticmethod
    def _validate_dataset_columns(dataset: Any, required_columns: List[str]) -> bool:
        """Проверяет физическое присутствие обязательных контрактов полей в стрелочной таблице."""
        try:
            columns = dataset.column_names
            missing = [col for col in required_columns if col not in columns]
            if missing:
                logger.error(f"❌ [FACTORY] В датасете отсутствуют обязательные колонки: {missing}")
                return False
            return True
        except Exception as e:
            logger.exception(f"❌ [FACTORY] Ошибка валидации колонок датасета: {str(e)}")
            return False

    @staticmethod
    def _ensure_chat_template(tokenizer: PreTrainedTokenizerFast) -> None:
        """Гарантирует наличие ChatML-шаблона у токенизатора для стабильного форматирования."""
        if tokenizer.chat_template is None:
            logger.warning("⚠️ [FACTORY] У токенизатора отсутствует chat_template. Принудительно выставляем ChatML для Qwen.")
            tokenizer.chat_template = LocalTrainerFactory.QWEN_CHAT_TEMPLATE

    @staticmethod
    def _estimate_max_seq_length(
        dataset: Any,
        tokenizer: PreTrainedTokenizerFast,
        max_limit: int = 512, # Безопасный предел VRAM для 7B модели на ноутбуке
        percentile: float = 0.95
    ) -> int:
        """LLMOps-анализ: сканирует датасет и вычисляет оптимальный размер окна по 95-му перцентилю."""
        logger.info("📊 [FACTORY] Оценка оптимальной длины последовательности на основе подмножества датасета...")
        LocalTrainerFactory._ensure_chat_template(tokenizer)
        try:
            sample_size = min(1000, len(dataset))
            sample = dataset.select(range(sample_size))
            lengths: List[int] = []           
            for row in sample:
                messages = [
                    {"role": "system", "content": row.get("instruction", "") or ""},
                    {"role": "user", "content": row.get("input", "") or ""},
                    {"role": "assistant", "content": row.get("output", "") or ""}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                tokens = tokenizer(text, truncation=False)["input_ids"]
                lengths.append(len(tokens))
            if not lengths:
                return 512                
            lengths.sort()
            idx = int(len(lengths) * percentile)
            estimated = lengths[min(idx, len(lengths) - 1)]
            estimated = min(estimated, max_limit)            
            logger.info(f"✅ [FACTORY] 95-й перцентиль длины токенов: {estimated}. Фиксируем max_seq_length={estimated}")
            return estimated
        except Exception as e:
            logger.warning(f"⚠️ [FACTORY] Не удалось оценить max_seq_length, откат к безопасному дефолту 512. Ошибка: {str(e)}")
            return 512

    @classmethod
    def create_trainer(
        cls,
        model: Any,
        tokenizer: PreTrainedTokenizerFast,
        dataset: Any, # Принимает полный DatasetDict из LocalDatasetHandler
        config: Any
    ) -> Result[SFTTrainer]:
        """Конфигурирует TrainingArguments и инициализирует аппаратно-оптимизированный SFTTrainer."""
        logger.info("⚙️ [TRAINER FACTORY] Инициализация аргументов Hugging Face Trainer...")       
        
        # Исправлено: Извлекаем сплиты train и test из входящего DatasetDict
        train_dataset = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
        eval_dataset = dataset["test"] if hasattr(dataset, "keys") and "test" in dataset else None
        
        required_columns: Final[List[str]] = ["instruction", "input", "output"]
        if not cls._validate_dataset_columns(train_dataset, required_columns):
            return Result.failure(
                error_message="Обучающий датасет не содержит обязательных колонок: instruction, input, output",
                code="DATASET_VALIDATION_FAILED"
            )
            
        cls._ensure_chat_template(tokenizer)
        response_template = cls._get_response_template(tokenizer)
        logger.info(f"🧩 [TRAINER FACTORY] Выбран response_template маскирования: '{response_template.replace('\n', '\\n')}'")
        
        # Вычисляем эффективное окно контекста на основе 95-го перцентиля
        effective_max_seq_length = config.max_seq_length or cls._estimate_max_seq_length(train_dataset, tokenizer)
        
        # Автоматически пробрасываем bfloat16 под архитектуру вашей RTX 5070 Ti
        supports_bf16: bool = torch.cuda.is_available() and torch.cuda.is_bf16_supported()       
        
        training_args = SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size if hasattr(config, 'per_device_train_batch_size') else 2,
            gradient_accumulation_steps=config.gradient_accumulation_steps if hasattr(config, 'gradient_accumulation_steps') else 4,
            num_train_epochs=config.num_train_epochs if hasattr(config, 'num_train_epochs') else 3,
            learning_rate=config.learning_rate if hasattr(config, 'learning_rate') else 2e-4,
            warmup_ratio=getattr(config, 'warmup_ratio', 0.03),
            # Вместо: warmup_ratio=getattr(config, 'warmup_ratio', 0.03),
            # Напишем явное количество шагов прогрева (warmup_steps). 
            # Оптимально для вашего датасета — 3-5 шагов:
            warmup_steps=5, 
            fp16=not supports_bf16,
            bf16=supports_bf16,
            logging_steps=getattr(config, 'logging_steps', 5),
            logging_strategy="steps",
            optim=getattr(config, 'optim', "paged_adamw_8bit"), # Экономия VRAM ноутбука
            weight_decay=getattr(config, 'weight_decay', 0.01),
            lr_scheduler_type=getattr(config, 'lr_scheduler_type', "cosine"),
            seed=getattr(config, 'seed', 42),
            output_dir=config.output_dir if hasattr(config, 'output_dir') else "./src/storage/lora_output",
            report_to="none",
            #evaluation_strategy="steps" if eval_dataset else "no",
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=10 if eval_dataset else None,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=1,
            disable_tqdm=False,
            # КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ДЛЯ ВЕРСИИ 0.13+:
            # Говорим TRL, что итоговый текст будет лежать в колонке "text"
            dataset_text_field="text",
            remove_unused_columns=False

            # ВАЖНО: Эти два параметра теперь живут строго внутри конфигурации!
            # max_seq_length=effective_max_seq_length,
            # packing=config.enable_packing if hasattr(config, 'enable_packing') else False
        )

        for attr_name in ["max_seq_length", "dataset_max_seq_length", "_max_seq_length"]:
            setattr(training_args, attr_name, effective_max_seq_length)
            
        for attr_name in ["packing", "dataset_packing", "_packing"]:
            setattr(training_args, attr_name, config.enable_packing if hasattr(config, 'enable_packing') else False)

        def formatting_prompts_func(examples: Any) -> List[str]:
            output_texts: List[str] = []
            
            # --- ОПРЕДЕЛЯЕМ ФОРМАТ ВХОДЯЩИХ ДАННЫХ (Батч или Одиночный элемент) ---
            # Если значения словаря — это списки, значит, рантайт старый (батчевый)
            is_batched = isinstance(next(iter(examples.values())), list) if examples else False

            if is_batched:
                # Старый синтаксис (для совместимости)
                num_elements = len(examples["input"])
                for i in range(num_elements):
                    instruction = examples["instruction"][i] or ""
                    user_input = examples["input"][i] or ""
                    output = examples["output"][i] or ""
                    
                    messages = [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": output}
                    ]
                    try:
                        text = tokenizer.apply_chat_template(messages, tokenize=False)
                        output_texts.append(text)
                    except Exception as e:
                        continue
            else:
                # СТРОГО ДЛЯ ВАШЕЙ НОВОЙ ВЕРСИИ TRL: Обработка одиночной строки
                instruction = examples.get("instruction", "") or ""
                user_input = examples.get("input", "") or ""
                output = examples.get("output", "") or ""
                
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": output}
                ]
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    # Новая TRL для построчного маппинга ждет либо одну строку, либо список из 1 строки
                    return text
                except Exception as e:
                    logger.warning(f"⚠️ [FACTORY] Ошибка ChatML-форматирования строки: {e}")
                    return ""

            return output_texts


        logger.info("📐 [TRAINER FACTORY] Активация Dynamic Padding DataCollator с маскированием контекста.")
        collator = CustomCompletionOnlyCollator(
            response_template=response_template,
            tokenizer=tokenizer
        )

        try:
            #trainer = SFTTrainer(
            #    model=model,
            #    processing_class=tokenizer,
            #    train_dataset=train_dataset,
            #    eval_dataset=eval_dataset, # ИСПРАВЛЕНО: Явно передаем тестовую выборку для расчета loss валидации
            #    formatting_func=formatting_prompts_func,
            #    data_collator=collator,
            #    max_seq_length=effective_max_seq_length,
            #    dataset_num_proc=None, # Однопоточный маппинг в WSL предотвращает мертвые блокировки (deadlocks)
            #    packing=config.enable_packing if hasattr(config, 'enable_packing') else False,
            #    args=training_args,
            #)
            # СБОРКА ТРЕНЕРА ЧЕРЕЗ СЕЙФ-СЛОВАРЬ (DYNAMIС KWARGS BINDING)

            # Формируем базовые железные аргументы
            trainer_kwargs = {
                "model": model,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "formatting_func": formatting_prompts_func,
                "data_collator": collator,
                "args": training_args,
            }
            
            # Динамически инжектируем токенизатор под старые/новые сигнатуры TRL
            if hasattr(SFTTrainer, "__init__") and "processing_class" in SFTTrainer.__init__.__code__.co_varnames:
                trainer_kwargs["processing_class"] = tokenizer
            else:
                trainer_kwargs["tokenizer"] = tokenizer

            # Динамически инжектируем max_seq_length strictly в тренера, если config его отторгает
            # trainer_kwargs["max_seq_length"] = effective_max_seq_length
            
            # Собираем инстанс
            trainer = SFTTrainer(**trainer_kwargs)
            
            logger.info("✅ [TRAINER FACTORY] SFTTrainer успешно сконфигурирован и собран!")
            return Result.success(trainer)

        except Exception as e:
            logger.exception("❌ [TRAINER FACTORY] Критический сбой инициализации фабрики")
            return Result.failure(
                error_message=f"Не удалось инициализировать SFTTrainer: {str(e)}",
                code="TRAINER_INIT_FAILED"
            )
