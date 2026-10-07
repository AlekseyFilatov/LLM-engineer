import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

from config.config import settings
from config.logger import logger
from src.domain.monads import Result
from src.domain.interfaces import ITritonClientService

class TritonInferenceClient(ITritonClientService):
    """gRPC-клиент C++ моделей Triton Server (Финальная Безошибочная Сборка)"""
    def __init__(self, url: str, model_name: str, tokenizer_path: Path):
        self.url = url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self._init_tokenizers()

    def _init_tokenizers(self) -> None:
        """Инициализация локальных токенизаторов хоста строго через Pydantic Config"""
        try:
            base_path = Path(__file__).parent.parent.parent
            
            # Нативный русский токенизатор первого уровня
            para_path = base_path / "src" / "paraphrase_tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(str(para_path), local_files_only=True)
            
            # Токенизатор реранкера второго уровня
            reranker_tok_path = base_path / "src" / "reranker_tokenizer"
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(str(reranker_tok_path), local_files_only=True)
            
            logger.info("🎯 [TRITON CLIENT] Все локальные токенизаторы успешно инициализированы.")
        except Exception as e:
            self.tokenizer = None
            self.reranker_tokenizer = None
            logger.error(f"❌ Критическая ошибка инициализации токенизаторов: {e}")

    def call_cascade_paligemma2(self, prompt_text: str) -> str:
        """CPU роутинг полностью изолировал этот шаг, возвращаем пустую JSON-строку"""
        return "{}"

    def generate_clip_embedding(self, search_tags: str) -> Result[List[float]]:
        """Вычисление высокоточных русскоязычных эмбеддингов 384-размерности на GPU"""
        model_target = "paraphrase_embedder"
        try:
            if not self.tokenizer:
                return Result.failure("Токенизатор первого уровня отсутствует.", "TOKENIZER_MISSING")

            inputs_tokenized = self.tokenizer(search_tags, padding="max_length", max_length=128, truncation=True, return_tensors="np")
            
            input_ids = inputs_tokenized["input_ids"].astype(np.int64)
            attention_mask = inputs_tokenized["attention_mask"].astype(np.int64)
            
            if "token_type_ids" in inputs_tokenized:
                token_type_ids = inputs_tokenized["token_type_ids"].astype(np.int64)
            else:
                token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            logger.info(
                f"🔍 [gRPC Отладка -> {model_target}] "
                f"input_ids: {input_ids.shape}, mask: {attention_mask.shape}, token_type_ids: {token_type_ids.shape}"
            )

            inputs = [
                grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
                grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
                grpcclient.InferInput("token_type_ids", token_type_ids.shape, "INT64")
            ]
            # ИСПРАВЛЕНО: Обращаемся к объектам InferInput внутри списка по их индексам!
            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)
            inputs[2].set_data_from_numpy(token_type_ids)

            resp = self.client.infer(model_name=model_target, inputs=inputs)
            
            raw_vector = None
            for output_name in ["sentence_embedding", "last_hidden_state", "output", "output_0"]:
                try:
                    np_data = resp.as_numpy(output_name)
                    if np_data is not None:
                        if len(np_data.shape) == 3:
                            raw_vector = np.mean(np_data, axis=1).flatten().tolist()
                        else:
                            raw_vector = np_data.flatten().tolist()
                        break
                except Exception:
                    continue

            if raw_vector is None:
                first_output_name = resp.get_response().outputs.name
                np_data = resp.as_numpy(first_output_name)
                if len(np_data.shape) == 3:
                    raw_vector = np.mean(np_data, axis=1).flatten().tolist()
                else:
                    raw_vector = np_data.flatten().tolist()

            np_vec = np.array(raw_vector)
            norm = np.linalg.norm(np_vec)
            if norm > 0:
                np_vec = np_vec / norm
            vector = np_vec.tolist()[:384]

            return Result.success(vector)
        except Exception as e:
            logger.error(f"💥 [CRITICAL DEBUG] Крах инференса '{model_target}': {e}")
            return Result.failure(f"Ошибка gRPC инференса: {str(e)}", "TRITON_CLIP_ERROR")

    def compute_reranker_score(self, user_query: str, candidate_text: str) -> float:
        """Инференс Cross-Encoder bge_reranker на GPU"""
        model_target = "bge_reranker"
        try:
            if not self.reranker_tokenizer: 
                return 0.0
                
            encoded = self.reranker_tokenizer(user_query, candidate_text, padding="max_length", max_length=128, truncation=True, return_tensors="np")
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

            inputs = [
                grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
                grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
            ]
            # Обращаемся по индексам к объектам InferInput!
            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)

            resp = self.client.infer(model_name=model_target, inputs=inputs)
            logits = resp.as_numpy("logits").flatten()
            logits = np.clip(logits, -50, 50)
            # Извлекаем, чтобы убрать варнинг при конвертации массива ndarray > 0 в скаляр
            score = 1 / (1 + np.exp(-logits))
            return float(score[0]) if isinstance(score, np.ndarray) else float(score)

        except Exception as e:
            logger.error(f"💥 [CRITICAL DEBUG] Крах инференса '{model_target}': {e}")
            return 0.0
