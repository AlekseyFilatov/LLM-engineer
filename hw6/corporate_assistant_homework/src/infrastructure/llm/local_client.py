import os
import torch
import asyncio
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict, Any

import json

from src.domain.interfaces import LLMInterface, CorporateAnswerSchema
from src.domain.monads import Result
from config.settings import settings

class LocalLLMClient(LLMInterface):
    """Промышленный клиент локальной LLM, полностью изолированный на диске ноутбука."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 [LOCAL LLM] Инициализация на устройстве: {self.device}")

        # Читаем токен авторизации строго из .env
        hf_token = settings.HF_TOKEN if settings.HF_TOKEN.strip() else None
        # Берем глобальный путь к кэшу на диске D
        cache_dir = os.getenv("HF_HOME", "./storage/models")

        if hf_token:
            print("🔑 [LOCAL LLM] Авторизация на Hugging Face Hub успешно активирована.")
        else:
            print("⚠️ [LOCAL LLM] HF_TOKEN не найден. Загрузка продолжится из публичного репозитория.")

        local_only = False
        if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
            # Если в папке уже что-то есть, запрещаем лезть в сеть
            local_only = True
            print("🔌 [LOCAL LLM] Локальный кэш обнаружен. Включение режима Offline-загрузки.")

        print(f"🤖 [LOCAL LLM] Загрузка токенизатора {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            cache_dir=cache_dir,
            local_files_only=local_only  # СТАЛО: Защита от повторного скачивания
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"🤖 [LOCAL LLM] Загрузка весов модели в кэш: {cache_dir}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            token=hf_token,
            cache_dir=cache_dir,
            local_files_only=local_only  # СТАЛО: Защита от повторного скачивания
        ).to(self.device)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print(f"🤖 [LOCAL LLM] Настройка конфигурации генерации...")
        self.gen_config = GenerationConfig(
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        print("✅ [LOCAL LLM] Модель успешно загружена и готова к работе в WSL!")

    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> Result[str]:
        """Синхронный монадический метод генерации ответа на основе найденного контекста."""
        try:
            # 1. Формируем текстовый блок контекста из найденных документов
            context_str = ""
            if context:
                context_str = "\n".join([
                    f"Документ: {doc.get('metadata', {}).get('source', 'Неизвестный')} "
                    f"(стр. {doc.get('metadata', {}).get('page', '1')})\n"
                    f"Текст: {doc.get('text', '')}\n---"
                    for doc in context
                ])

            # 2. Вытаскиваем системный промпт из нашего YAML через settings
            system_prompt = settings.corporate_prompt
            
            # Подмешиваем контекст к вопросу пользователя
            user_message_with_context = f"Контекст из корпоративной базы документов:\n{context_str}\n\nВопрос сотрудника: {prompt}"

            # 3. Собираем структуру сообщений для чат-шаблона Qwen
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_with_context}
            ]
            
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)

            # 4. Локальный инференс весов
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, generation_config=self.gen_config)

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            decoded_list = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            raw_string_response = str(decoded_list[0]) if decoded_list else ""
            
            # Возвращаем чистый успешный результат, обернутый в монаду
            return Result.success(raw_string_response.strip())

        except Exception as e:
            # Перехватываем Out-Of-Memory (OOM) на GPU или любые другие сбои инференса
            return Result.failure(
                error_message=f"Сбой локального инференса LLM (Transformers): {str(e)}",
                code="LLM_INFERENCE_ERROR"
            )

    async def generate_response_async(self, prompt: str, context: List[Dict[str, Any]], custom_system: str = None) -> Result[str]:
        # Оборачиваем тяжелый инференс в поток, чтобы WSL-сервер не фризился
        def _sync_generation():
            try:
                context_str = ""
                if context:
                    context_str = "\n".join([
                        f"Документ: {doc.get('metadata', {}).get('source', 'Неизвестный')} (стр. {doc.get('metadata', {}).get('page', '1')})\nТекст: {doc.get('text', '')}\n---"
                        for doc in context
                    ])

                system_prompt = custom_system if custom_system else settings.corporate_prompt
                user_message_with_context = f"Контекст из корпоративной базы документов:\n{context_str}\n\nВопрос сотрудника: {prompt}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_with_context}
                ]

                text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, generation_config=self.gen_config)

                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                decoded_list = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                return Result.success(str(decoded_list[0]).strip() if decoded_list else "")
            except Exception as e:
                return Result.failure(f"Ошибка инференса: {str(e)}", "LLM_THREAD_ERROR")

        return await asyncio.to_thread(_sync_generation)

    async def generate_structured_response_async(self, prompt: str, context: list, custom_system: str) -> Result[CorporateAnswerSchema]:
        """Генерация строго структурированного JSON-ответа с защитой от лишних символов и текстовых отказов."""
        try:
            json_example = (
                "{\n"
                '  "text_answer": "Текст ответа сотруднику на русском языке...",\n'
                '  "citations": [{"source": "имя_файла.pdf", "page": "номер_или_стр"}],\n'
                '  "confidence_score": 0.95,\n'
                '  "needs_human_review": false\n'
                "}"
            )
            
            structured_system_prompt = (
                f"{custom_system}\n"
                f"Ты обязан вернуть ответ СТРОГО в формате JSON. Вот точный образец структуры, которой ты должен следовать:\n"
                f"{json_example}\n"
                f"Внимание: Не пиши ничего, кроме чистого JSON-объекта. Не добавляй никаких пояснений до или после JSON. "
                f"Не оборачивай JSON в маркдаун блоки ```json ... ```."
            )

            context_str = ""
            if context:
                context_str = "\n".join([
                    f"Документ: {doc.get('metadata', {}).get('source', 'Неизвестный')} (стр. {doc.get('metadata', {}).get('page', '1')})\nТекст: {doc.get('text', '')}\n---"
                    for doc in context
                ])
            
            user_message = f"Контекст:\n{context_str}\n\nВопрос: {prompt}"

            def _sync_generation():
                messages = [
                    {"role": "system", "content": structured_system_prompt},
                    {"role": "user", "content": user_message}
                ]
                text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, generation_config=self.gen_config)

                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                decoded_list = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # ИСПРАВЛЕНО: Безопасное извлечение строки из батча
                raw_json_str = decoded_list[0].strip() if decoded_list else "{}"
                
                # 1. Избавляемся от markdown-оберток
                raw_json_str = re.sub(r"^```json\s*|```$", "", raw_json_str, flags=re.MULTILINE).strip()
                
                # --- УМНЫЙ СЕРАНТИЧЕСКИЙ ФИЛЬТР ТЕКСТОВЫХ ОТКАЗОВ ---
                refusal_markers = ["я не знаю", "нет данных", "отсутствует информация", "не могу ответить", "извините"]
                is_refusal = any(marker in raw_json_str.lower() for marker in refusal_markers)

                # ИСПРАВЛЕНО: Проверяем на сырой текст ДО того, как принудительно обрежем скобки
                if not raw_json_str.startswith("{"):
                    if is_refusal:
                        print("🛑 [Infrastructure Guard] Корректный перехват текстового отказа. Авто-упаковка в JSON.")
                        validated_data = CorporateAnswerSchema(
                            text_answer="Я не знаю. В корпоративной базе знаний нет документов, регламентирующих данный вопрос.",
                            citations=[],
                            confidence_score=0.0,
                            needs_human_review=True
                        )
                        return Result.success(validated_data)
                    else:
                        return Result.failure(
                            error_message=f"Модель нарушила формат JSON и выдала сырой текст: '{raw_json_str[:50]}...'",
                            code="INVALID_TEXT_RESPONSE_FORMAT"
                        )

                # 2. Если это JSON (начинается с '{'), делаем хирургическую обрезку лишних символов (trailing characters)
                start_idx = raw_json_str.find('{')
                end_idx = raw_json_str.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    raw_json_str = raw_json_str[start_idx:end_idx + 1]

                # Стандартный парсинг очищенного JSON
                validated_data = CorporateAnswerSchema.model_validate_json(raw_json_str)
                return Result.success(validated_data)

            # Запускаем поток и возвращаем полученный Result (Success или Failure)
            return await asyncio.to_thread(_sync_generation)

        except Exception as e:
            # Сюда код прилетит только если упал сам токенизатор или PyTorch (например, CUDA OOM)
            return Result.failure(f"Ошибка валидации структурированного JSON ответа: {str(e)}", "JSON_STRUCTURE_ERROR")
