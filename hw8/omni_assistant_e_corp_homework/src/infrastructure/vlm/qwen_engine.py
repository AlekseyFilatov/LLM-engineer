import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.domain.interfaces import BaseVLMEngine
from src.domain.monads import Result
from config.logger import logger
from config.settings import settings

class QwenVLMEngine(BaseVLMEngine):
    """
    Высокоскоростной оффлайн-движок компьютерного зрения и рассуждений на базе Qwen2-VL-2B.
    Нативно работает в bfloat16/float16 на тензорных ядрах.
    """
    def __init__(self) -> None:
        self.model_dir: Path = Path(settings.VLM_DIR)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._initialized = False

        # Метрики LLMOps
        self._total_generations: int = 0
        self._latency_history: List[float] = []

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        logger.info("=== [VLM] Инициализация оффлайн-модели Qwen2-VL-2B ===")
        try:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Директория весов VLM не найдена: {self.model_dir}")
                
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"⚡ [VLM] Выбран тип данных вычислений: {compute_dtype}")

            # Загружаем процессор изображений и текста
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            )

            # Загружаем базовую архитектуру зрения и текста
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(self.model_dir),
                torch_dtype=compute_dtype,
                device_map="auto" if self.device == "cuda" else "cpu",
                local_files_only=True
            )
            self._initialized = True
            logger.info("✅ [VLM INITIALIZATION] Qwen2-VL успешно развернута в ОЗУ/VRAM.")
        except Exception as e:
            logger.exception("❌ [VLM INITIALIZATION] Критический крах загрузки VLM: %s", e)
            raise RuntimeError(f"Ошибка инициализации Qwen2-VL: {e}") from e

    def generate(self, image_path: Path, prompt_text: str) -> Result[str]:
        self._total_generations += 1
        self._lazy_init()

        if not image_path.exists():
            return Result.failure(f"Изображение не найдено по пути: {image_path}", "VLM_IMAGE_NOT_FOUND")

        try:
            logger.info("👁️ [VLM] Запуск фазы визуального анализа контекста...")
            t_start = time.perf_counter()

            # Строгий промпт, подготавливающий текст для последующего TTS (без списков и разметки)
            system_prompt = settings.VLM_SYSTEM_PROMPT

            # ИСПРАВЛЕНО: Форматирование текста внутри сообщений склеено без синтаксического разрыва строки
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": f"{system_prompt}\n\nВопрос пользователя: {prompt_text}"}
                    ]
                }
            ]

            # Предобработка текста через шаблонизатор
            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Извлекаем и масштабируем геометрические патчи картинки силами qwen_vl_utils
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Упаковываем всё в тензоры PyTorch для GPU
            inputs = self._processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Контролируемый инференс без градиентов с KV-кэшированием
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=settings.MAX_NEW_TOKENS,
                    use_cache=True,
                    do_sample=False,  # Отключаем сэмплирование для стабильности ответов
                    repetition_penalty=1.2,
                    # ДОБАВЛЕНО: Явные токены остановки, защищающие от бесконечного зацикливания
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                    eos_token_id=self._processor.tokenizer.eos_token_id
                )
            
            # Отсекаем токены входного промпта, оставляя только чистый ответ модели
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)
            ]
            
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            # Дополнительная санитария: убираем случайные переносы строк и спецсимволы
            output_text = output_text.replace("*", "").replace("#", "").strip()
            output_text = " ".join(output_text.split())  # схлопываем лишние пробелы

            latency = time.perf_counter() - t_start
            self._latency_history.append(latency)
            
            logger.info(f"✅ [VLM SUCCESS] Визуальный ответ сгенерирован за {latency:.3f} сек.")
            logger.debug(f"   Ответ ИИ: '{output_text}'")
            
            return Result.success(output_text)

        except Exception as e:
            logger.exception("❌ [VLM CRASH] Непредвиденный сбой тензорных вычислений Qwen2-VL: %s", e)
            return Result.failure(f"Крах VLM инференса: {str(e)}", "VLM_RUNTIME_ERROR")

    def get_telemetry(self) -> Dict[str, Any]:
        avg_latency = float(torch.tensor(self._latency_history).mean()) if self._latency_history else 0.0
        return {
            "total_generations": self._total_generations,
            "avg_latency_seconds": round(avg_latency, 3),
            "device": self.device
        }
