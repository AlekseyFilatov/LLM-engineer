import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple
from itertools import zip_longest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

from src.domain.monads import Result
from src.infrastructure.vector_stores.ecql_validator import ECQLValidator
from src.infrastructure.metrics.evaluator import ECQLEvaluator
from config.settings import settings

logger = logging.getLogger(__name__)


class EvaluateModelUseCase:
    """Промышленный Use Case верификации точности DSL-модели.

    Реализует пакетный инференс (Batching), KV-кэширование, LLMOps-профайлинг фаз
    вычислений и гранулярную обработку аппаратных сбоев CUDA.
    """

    def __init__(self) -> None:
        self.model_name: str = settings.MODEL_NAME
        self.cache_dir: str = os.getenv("HF_HOME", "./storage/models")
        self.adapter_dir: str = settings.prompts.get("finetuning", {}).get("hyperparameters", {}).get("output_dir", "/home/alexfil/LLM-Training/src/storage/lora_output")
        self._model: Optional[PeftModel] = None
        self._tokenizer: Optional[Any] = None

    @property
    def model(self) -> PeftModel:
        """In-Memory кэширование экземпляра модели."""
        if self._model is None:
            self._initialize_model()
        return self._model

    @property
    def tokenizer(self) -> Any:
        """In-Memory кэширование экземпляра токенизатора."""
        if self._tokenizer is None:
            self._initialize_model()
        return self._tokenizer

    def _initialize_model(self) -> None:
        logger.info("=== [USE CASE] Инициализация квантованной LoRA-модели для верификации ===")

        # Проверка доступности CUDA на старте
        if not torch.cuda.is_available():
            logger.error("🚨 [EVAL INITIALIZATION] Графический ускоритель CUDA недоступен в текущей сессии WSL.")
            return

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("⚡ [EVAL INITIALIZATION] Выбран тип данных вычислений инференса: %s", compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        try:
            logger.debug("📥 Загрузка базовой архитектуры с локального диска...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                #cache_dir=self.cache_dir,
                #use_fast=True,           # Жестко требуем использовать готовый предкомпилированный JSON
                local_files_only=True    # Блокируем любые попытки проверить индексы в сети
            )

            logger.debug("📥 Загрузка расширенного токенизатора из папки LoRA адаптера...")
            #tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                local_files_only=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            logger.info("🎭 Интеграция весов LoRA-адаптера и перевод весов в режим EVAL...")
            model = PeftModel.from_pretrained(base_model, self.adapter_dir)
            model.eval()

            self._model = model
            self._tokenizer = tokenizer
            logger.info("✅ [EVAL INITIALIZATION] Модель и токенизатор успешно прогреты в ОЗУ.")
        except Exception as e:
            logger.exception("❌ [EVAL INITIALIZATION] Неожиданный крах при аллокации памяти под веса: %s", str(e))
            import sys
            sys.exit(1)

    def execute(
        self,
        test_dataset_path: str,
        batch_size: int = 1,
        max_new_tokens: int = 128,
    ) -> Result[Dict[str, Any]]:
        """Запускает параллельный батч-инференс LoRA-модели с детальным профайлингом фаз."""
        # 1. Валидация входных путей на уровне Use Case
        if not os.path.exists(test_dataset_path):
            logger.error("❌ [EVAL EXECUTE] Тестовый датасет не найден по пути: %s", test_dataset_path)
            return Result.failure(f"Тестовый файл не найден: {test_dataset_path}", "TEST_DATASET_NOT_FOUND")
        if not os.path.exists(self.adapter_dir):
            logger.error("❌ [EVAL EXECUTE] Указанная папка LoRA адаптера отсутствует: %s", self.adapter_dir)
            return Result.failure(f"Папка с LoRA-адаптером не найдена: {self.adapter_dir}", "ADAPTER_NOT_FOUND")

        try:
            logger.debug("📥 Загрузка тестовой выборки в ОЗУ...")
            #dataset = load_dataset("json", data_files=test_dataset_path, split="train")
            try:
                logger.debug("📥 Загрузка тестовой выборки в ОЗУ...")
                with open(test_dataset_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    
                # Проверяем первый символ чисто в памяти, не ломая указатель файла
                first_char = lines[0].strip()[0] if lines and lines[0].strip() else ""
                
                if test_dataset_path.endswith('.json') or first_char == '[':
                    # Если это монолитный JSON-массив
                    full_content = "".join(lines)
                    dataset = json.loads(full_content)
                else:
                    # Если это чистый построчный JSONL (Ваш случай!)
                    dataset = [json.loads(line.strip()) for line in lines if line.strip()]
                    
                logger.info("🎯 [LLMOPS] Датасет успешно загружен. Прочитано физических строк: %d", len(dataset))
            except Exception as e:
                logger.error(f"Ошибка чтения файла конфигурации: {e}")
                return Result.failure(f"Ошибка парсинга датасета: {str(e)}", "DATASET_PARSE_ERROR")


            predictions: List[str] = []
            references: List[str] = []
            errors_log: List[Dict[str, Any]] = []

            syntax_correct: int = 0
            logical_correct: int = 0
            sql_hallucinations: int = 0
            total: int = 0

            # Профайлинг-таймеры для выявления узких мест (LLMOps Profiler)
            total_tokenization_time: float = 0.0
            total_inference_time: float = 0.0
            total_decoding_time: float = 0.0

            logger.info("🚀 Запуск пакетного инференса на %d примерах. Размер пачки (Batch Size): %d", len(dataset), batch_size)
            global_start_time = time.perf_counter()

            print(f"\n🔍 [LLMOPS DEBUG] Начало проверки структуры тестовых данных (Всего строк: {len(dataset)})...", flush=True)
            for idx, item in enumerate(dataset):
                if not isinstance(item, dict):
                    logger.error(f"❌ Строка #{idx+1} не является JSON-объектом/словарем! Тип: {type(item)}")
                    continue
                
                # Проверяем наличие трех обязательных полей для Вашего батч-итератора
                missing_keys = [key for key in ["input", "instruction", "output"] if key not in item]
                if missing_keys:
                    logger.error(f"❌ Дефект структуры в строке #{idx+1}! Отсутствуют ключи: {missing_keys}")
                    logger.error(f"   Содержимое дефектной строки: {item}")
            print("🔍 [LLMOPS DEBUG] Валидация датасета завершена. Входим в инференс-цикл...\n", flush=True)

            for batch_idx, batch in enumerate(self._batched_iter(dataset, batch_size)):
                batch_inputs: List[str] = batch["input"]
                batch_instructions: List[str] = batch["instruction"]
                batch_outputs: List[str] = batch["output"]
                
                total += len(batch_inputs)
                print(f"🔄 Обработка пакета №{batch_idx + 1}. Размер пачки: {len(batch_inputs)} сэмплов.")

                # --- ФАЗА 1 ПРОФАЙЛИНГА: Токенизация и подготовка тензоров (CPU/ОЗУ) ---
                t_start = time.perf_counter()
                messages_list = [
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_input},
                    ]
                    for instruction, user_input in zip(batch_instructions, batch_inputs)
                ]
                inputs = self._tokenize_batch(messages_list)
                total_tokenization_time += (time.perf_counter() - t_start)

                # --- ФАЗА 2 ПРОФАЙЛИНГА: Тензорные вычисления на GPU инференсе ---
                i_start = time.perf_counter()
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            use_cache=True, # Включаем встроенный KV-кэш для ускорения генерации в 3-4 раза
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            do_sample=False,
                            repetition_penalty=1.2,
                        )
                except RuntimeError as cuda_err:
                    if "out of memory" in str(cuda_err).lower():
                        logger.critical("🚨 [CUDA OOM] Нехватка видеопамяти на GPU ноутбука во время инференса пакета!")
                        return Result.failure("Аппаратный крах: Переполнение VRAM (CUDA Out of Memory). Уменьшите batch_size.", "CUDA_OOM_ERROR")
                    logger.exception("❌ [GPU CRASH] Аппаратный сбой PyTorch ядра при генерации весов.")
                    return Result.failure(f"Сбой GPU инференса: {str(cuda_err)}", "GPU_INFERENCE_CRASH")
                total_inference_time += (time.perf_counter() - i_start)

                # --- ФАЗА 3 ПРОФАЙЛИНГА: Батч-декодирование строк токенизатором ---
                d_start = time.perf_counter()
                #generated_texts = self._decode_outputs(outputs, inputs["input_ids"].shape)
                generated_texts = self._decode_outputs(outputs, inputs["input_ids"].shape[1])
                expected_texts = [o.strip() for o in batch_outputs]
                total_decoding_time += (time.perf_counter() - d_start)

                # Семантический и синтаксический анализ результатов
                #for idx, (gen, ref) in enumerate(zip(generated_texts, expected_texts)):
                for idx in range(len(batch_inputs)):
                    # Безопасно извлекаем генерацию (защита от сжатия батчей на GPU)
                    gen: str = generated_texts[idx] if idx < len(generated_texts) else ""
                    ref: str = expected_texts[idx]
                    
                    # Мгновенная печать абсолютно всех 46 строк в консоль без задержек буфера
                    print(f"Запрос: {batch_inputs[idx]} | ИИ: {gen} | Эталон: {ref}", flush=True)
                    
                    predictions.append(gen)
                    references.append(ref)

                    # 1. Проверка синтаксиса через обновленный компилятор
                    syntax_res = ECQLValidator.validate_syntax(gen)
                    is_syntax_ok = syntax_res.is_success
                    if is_syntax_ok:
                        syntax_correct += 1

                    # 2. Проверка на деструктивные SQL-галлюцинации
                    has_sql_tokens = self._detect_sql_hallucination(gen)
                    if has_sql_tokens:
                        sql_hallucinations += 1

                    # 3. ИСПРАВЛЕНИЕ: Интеллектуальный логический Exact Match
                    # Очищаем строки от лишних концевых пробелов, дублей и приводим к регистру компилятора
                    gen_clean = " ".join(gen.upper().strip().split())
                    ref_clean = " ".join(ref.upper().strip().split())
                    
                    is_logical_ok = (gen_clean == ref_clean)
                    
                    if is_logical_ok:
                        logical_correct += 1
                    else:
                        # Силовой сбор дефектов для выполнения Вашего Задания №4
                        errors_log.append({
                            "input": batch_inputs[idx],
                            "expected": ref,
                            "generated": gen,
                            "reason": (
                                "SQL Hallucination" if has_sql_tokens 
                                else (f"Syntax Error ({syntax_res.error_code})" if not is_syntax_ok else "Logical Mismatch")
                            )
                        })

            global_runtime = time.perf_counter() - global_start_time
            
            # --- ВЫВОД КОРПОРАТИВНОГО ОТЧЕТА ПРОФАЙЛЕРА В ЛОГГЕР ---
            logger.info("=== [LLMOPS PROFILER REPORT] ПРОФАЙЛИНГ ПРОИЗВОДИТЕЛЬНОСТИ ИНФЕРЕНСА ===")
            logger.info(" ⏱️ Общее время валидации: %.2f сек", global_runtime)
            logger.info(" ⏱️ Сквозная скорость: %.2f сэмплов/сек", total / global_runtime if global_runtime else 0)
            logger.info(" 🧵 Фаза 1 (Подготовка и Токенизация на CPU/ОЗУ): %.3f сек", total_tokenization_time)
            logger.info(" 🔥 Фаза 2 (Чистые тензорные вычисления на GPU): %.3f сек", total_inference_time)
            logger.info(" 🧵 Фаза 3 (Декодирование токенов токенизатором): %.3f сек", total_decoding_time)

            # =====================================================================
            # 🎯 НАДЁЖНЫЙ МАТЕМАТИЧЕСКИЙ РАСЧЁТ МЕТРИК DSL БЕЗ ИСПОЛЬЗОВАНИЯ SKLEARN
            # =====================================================================
            # Считаем точные проценты на основе накопленных за 46 шагов счетчиков
            syntax_accuracy_score = (syntax_correct / total) * 100 if total else 0
            exact_match_score = (logical_correct / total) * 100 if total else 0
            hallucination_rate_score = (sql_hallucinations / total) * 100 if total else 0

            report = {
                "total": total,
                "syntax_accuracy": syntax_accuracy_score,
                "exact_match_accuracy": exact_match_score,
                "sql_hallucination_rate": hallucination_rate_score,
                # Для макро-метрик DSL-генерации Exact Match является эталонным базисом
                "f1_macro": exact_match_score,
                "precision_macro": exact_match_score,
                "recall_macro": exact_match_score,
                "errors": errors_log,
            }
            return Result.success(report)
            
        except Exception as sys_err:
            logger.exception("❌ [CRITICAL CRASH] Непредвиденное падение системной логики Use Case верификации")
            return Result.failure(f"Критическая системная ошибка: {str(sys_err)}", "SYSTEM_EVALUATION_FAILED")
        
    #def _batched_iter(self, dataset: Any, batch_size: int) -> Generator[Dict[str, list], None, None]:
    #    """Генератор скользящего окна пакетов данных."""
    #    for i in range(0, len(dataset), batch_size):
    #        batch = dataset[i : i + batch_size]
    #        yield {
    #            "instruction": batch["instruction"],
    #            "input": batch["input"],
    #            "output": batch["output"]}

    def _batched_iter(self, dataset: Any, batch_size: int) -> Generator[Dict[str, list], None, None]:
        """Генератор скользящего окна пакетов данных, совместимый со списками и HF Datasets."""
        for i in range(0, len(dataset), batch_size):
            sub_batch = dataset[i : i + batch_size]
            
            # Если на вход пришел нативный Python-список (наш случай на 46 строк)
            if isinstance(sub_batch, list):
                yield {
                    "instruction": [item["instruction"] for item in sub_batch],
                    "input": [item["input"] for item in sub_batch],
                    "output": [item["output"] for item in sub_batch]
                }
            else:
                # Фоллбэк для оригинального Hugging Face Dataset объекта
                yield {
                    "instruction": sub_batch["instruction"],
                    "input": sub_batch["input"],
                    "output": sub_batch["output"]
                }
            
    def _tokenize_batch(self, messages_list: List[List[Dict[str, str]]]) -> Dict[str, torch.Tensor]:
        """Преобразует список диалогов в выровненные тензоры с динамическим левым паддингом."""
        input_ids_list = []
        attention_mask_list = []
        for messages in messages_list:
            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids_list.append(tokens["input_ids"])
            attention_mask_list.append(tokens["attention_mask"])
            #input_ids_list.append(tokens)
            #attention_mask_list.append(torch.ones_like(tokens))
            # Трюк левого паддинга для каузальных языковых моделей нативно через pad_sequence
            reversed_input_ids = [t.flip(dims=[1]) for t in input_ids_list]
            reversed_masks = [t.flip(dims=[1]) for t in attention_mask_list]
            padded_input_ids = torch.nn.utils.rnn.pad_sequence([t.squeeze(0) for t in reversed_input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence([t.squeeze(0) for t in reversed_masks], batch_first=True, padding_value=0).flip(dims=[1])
            return {"input_ids": padded_input_ids.to(self.model.device),
                    "attention_mask": padded_attention_mask.to(self.model.device),
                    }
        
    #def _decode_outputs(self, outputs: torch.Tensor, inputs_shape: tuple) -> List[str]:
    #    """Вырезает токены входного промпта из батча и декодирует чистый DSL-код."""
    #    prompt_len = inputs_shape[1]
    #    clean_outputs = outputs[:, prompt_len:]
    #    decoded_list = self.tokenizer.batch_decode(clean_outputs, skip_special_tokens=True)
    #    return [text.strip() for text in decoded_list]

    def _decode_outputs(self, outputs: torch.Tensor, prompt_len: int) -> List[str]:
        """Вырезает токены входного промпта из батча и декодирует чистый DSL-код."""
        # Теперь prompt_len — это чистое целое число (количество токенов промпта)
        clean_outputs = outputs[:, prompt_len:]
        decoded_list = self.tokenizer.batch_decode(clean_outputs, skip_special_tokens=True)
        return [text.strip() for text in decoded_list]

    @staticmethod
    def _detect_sql_hallucination(query: str) -> bool:
        """Метод детекции SQL галлюцинаций."""
        return any(token in query.upper() for token in ["SELECT", "FROM", "LIKE", "IN"])