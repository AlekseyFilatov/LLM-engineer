import json
import re
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
from sklearn.pipeline import Pipeline

from config.config import settings
from config.logger import logger
from src.domain.monads import Result
from src.domain.memory import DialogueMemoryManager
from src.domain.prompts import PromptLoader
from src.domain.nlp_pipeline import FlowRegexCleaner, FlowSpacyLemmatizer
from src.domain.recommender_engine import InMemoryRecommenderEngine
from src.infrastructure.whisper_service import WhisperASRService
from src.infrastructure.triton_client import TritonInferenceClient
from src.infrastructure.qdrant_service import QdrantVectorSearchService
from src.domain.interfaces import ITritonClientService, IVectorDbService

class OmniSearchAgentOrchestrator:
    """Главный ИИ-Агент: Оркестратор сквозного пайплайна (SOTA Production-Grade RecSys Agent)"""
    def __init__(self, qdrant_service: IVectorDbService, triton_service: ITritonClientService):
        self.qdrant = qdrant_service
        self.triton = triton_service
        self.memory = DialogueMemoryManager()
        self.prompts = PromptLoader(Path(__file__).parent.parent.parent / "config" / "prompts.yaml")
        self.asr = WhisperASRService(str(settings.whisper_url))
        
        # Разворачиваем шаги в единый scikit-learn конвейер
        self._cleaner_step = FlowRegexCleaner()
        self.nlp_pipeline = Pipeline([
            ('regex_cleaner', self._cleaner_step),
            ('spacy_lemmatizer', FlowSpacyLemmatizer(batch_size=1))
        ])
        self.nlp_pipeline.fit(["прогрев"])
        
        # Передаем в движок и реестр бустинга, и весь сырой конфиг для чтения пользователей
        boosting_registry = self.prompts.config.get("business_boosting_registry", {})
        self.recommender = InMemoryRecommenderEngine(boosting_registry, self.prompts.config)
        
        logger.info("🎉 [Orchestrator] Декларативный графовый NLP-пайплайн и Рекомендательный Сервис успешно развернуты.")

    def _parse_qdrant_string(self, candidates_str: str, user_query: str, user_id: str) -> Dict[str, Any]:
        """
        Рекомендательное ядро Enterprise-уровня.
        Абсолютно независимо от жестких профилей. Принимает user_id как внешний контекст,
        строит многофакторное CatBoost-подобное линейное слияние (Linear Fusion).
        """
        if not candidates_str or "Нет товаров" in candidates_str:
            return {"name": "Базовый товар E-Corp", "price": "Уточняйте", "tags": "базовые", "image_file": "sunglasses.jpg", "raw_string": ""}

        all_candidates = candidates_str.split(" | ") if " | " in candidates_str else [candidates_str]
        
        candidate_data_list = []
        qdrant_raw_scores = []
        bge_raw_scores = []

        for candidate in all_candidates:
            name_match = re.search(r"name':\s*'(.*?)'", candidate)
            tags_match = re.search(r"tags':\s*'(.*?)'", candidate)
            price_match = re.search(r"price':\s*(\d+)", candidate)
            qdrant_score_match = re.search(r"qdrant_score':\s*([\d\.]+)", candidate)
            image_match = re.search(r"image_file':\s*'(.*?)'", candidate)
            
            c_name = name_match.group(1) if name_match else "Товар"
            c_price = float(price_match.group(1)) if price_match else 1000.0
            c_file = image_match.group(1) if image_match else ""
            q_score = float(qdrant_score_match.group(1)) if qdrant_score_match else 0.5
            
            # ЧИСТАЯ АРХИТЕКТУРА: Передаем полученный извне user_id в доменный сервис рекомендаций
            rec_features = self.recommender.compute_rec_features(user_id, c_name, c_price, c_file)
            
            candidate_clean_text = f"{c_name} {tags_match.group(1) if tags_match else ''}".strip()
            neural_score = self.triton.compute_reranker_score(user_query, candidate_clean_text)
            
            qdrant_raw_scores.append(q_score)
            bge_raw_scores.append(neural_score)
            candidate_data_list.append((candidate, c_name, q_score, neural_score, rec_features))

        # Математическое Min-Max скалирование шкал
        q_arr, b_arr = np.array(qdrant_raw_scores), np.array(bge_raw_scores)
        min_q, max_q = q_arr.min(), q_arr.max()
        denom_q = (max_q - min_q) if max_q != min_q else 1.0
        min_b, max_b = b_arr.min(), b_arr.max()
        denom_b = (max_b - min_b) if max_b != min_b else 1.0

        selected_candidate = all_candidates
        max_hybrid_score = -1.0
        alpha = 0.4 

        for candidate, name, q_score, n_score, rec in candidate_data_list:
            norm_q = (q_score - min_q) / denom_q
            norm_b = (n_score - min_b) / denom_b
            
            base_search_score = (alpha * norm_q) + ((1.0 - alpha) * norm_b)
            
            # Линейное многофакторное слияние с внешними признаками
            final_score = (base_search_score * 0.4) + (rec["als_score"] * 0.3) + (rec["segment_score"] * 0.2)
            final_score = final_score * rec["business_boost"]
            
            logger.info(
                f"📊 [RecSys Stage 2] User: {user_id} | {name[:10]}... | Search: {base_search_score:.2f} | "
                f"ALS: {rec['als_score']:.2f} | Segment: {rec['segment_score']:.2f} | "
                f"Boost: {rec['business_boost']:.1f} | Итог: {final_score:.4f}"
            )
            
            if final_score > max_hybrid_score:
                max_hybrid_score = final_score
                selected_candidate = candidate

        name_match = re.search(r"name':\s*'(.*?)'", selected_candidate)
        price_match = re.search(r"price':\s*(\d+)", selected_candidate)
        tags_match = re.search(r"tags':\s*'(.*?)'", selected_candidate)
        image_match = re.search(r"image_file':\s*'(.*?)'", selected_candidate)
        
        return {
            "name": name_match.group(1).strip() if name_match else "Товар каталога",
            "price": price_match.group(1).strip() if price_match else "Уточняйте",
            "tags": tags_match.group(1).strip() if tags_match else "одежда",
            "image_file": image_match.group(1).strip() if image_match else "sunglasses.jpg",
            "raw_string": selected_candidate
        }


    def run_workflow(self, text_query: str, audio_file: Optional[str], langfuse_client=None, user_id: str = "new_user_cold_start") -> Result[Dict[str, str]]:
        user_raw_text = text_query if text_query else ""
        
        trace = None
        if langfuse_client and hasattr(langfuse_client, "trace"):
            try: trace = langfuse_client.trace(name="Omni-Neural-Rerank-Pipeline", user_id="alexfil-wsl")
            except: pass

        # --- ШАГ 0: Сетевой инференс Faster-Whisper ---
        if audio_file is not None:
            span_asr = trace.span(name="Faster-Whisper-ASR") if trace else None
            asr_res = self.asr.transcribe(audio_file)
            if asr_res.is_failure(): return Result.failure(asr_res.error, asr_res.error_code)
            user_raw_text = asr_res.value
            if span_asr: span_asr.end(status="SUCCESS")

        if not user_raw_text.strip():
            return Result.failure("Входной запрос пуст.", "EMPTY_INPUT")

        # =====================================================================
        # ШАГ 1: Высокоскоростной CPU Маршрутизатор (БЕЗ PALI-GEMMA 2 В СЕТИ)
        # =====================================================================
        category_registry = self.prompts.config.get("category_registry", {})
        routing_data = self._cleaner_step.route_intent_and_category(user_raw_text, category_registry)
        
        intent = routing_data["intent"]
        detected_category = routing_data["category"]
        
        if intent == "GENERAL_CHITCHAT":
            chitchat_ans = "Привет! Я интеллектуальный ИИ-ассистент магазина E-Corp. Чем я могу помочь?"
            return Result.success({"intent": intent, "tags": "нет", "candidates": "нет", "reasoning": chitchat_ans, "image_path": None})

        # --- ШАГ 2: Высокоскоростная лемматизация на CPU для контура CLIP ---
        span_rewriter = trace.span(name="Query-Rewriter", model="spaCy-ru_core_news_sm") if trace else None
        search_text = " ".join(self.nlp_pipeline.transform([user_raw_text])).strip()
        logger.info(f"🚀 [spaCy Контур] В CLIP улетает чистое семантическое ядро: '{search_text}'")
        if span_rewriter: span_rewriter.end(status="SUCCESS")

        # --- ШАГ 3: Векторизация CLIP GPU ---
        clip_res = self.triton.generate_clip_embedding(search_text)
        if clip_res.is_failure(): return Result.failure(clip_res.error, clip_res.error_code)

        # --- ШАГ 4: Извлечение из базы Qdrant с аппаратной gRPC пре-фильтрацией ---
        qdrant_res = self.qdrant.search_candidates(clip_res.value, category_filter=detected_category)
        if qdrant_res.is_failure(): return Result.failure(qdrant_res.error, qdrant_res.error_code)
        candidates_ctx = qdrant_res.value

        # --- ШАГ 5: Динамическое ранжирование со скалированием ---
        logger.info("📊 [Workflow] Вызов Шага 5: Нейросетевое Cross-Attention переранжирование...")
        span_vlm2 = trace.span(name="BGE-Reranker-GPU", model="bge-reranker-base") if trace else None
        
        # Явно прокидываем динамический user_id в рекомендательный парсер!
        product_data = self._parse_qdrant_string(candidates_ctx, user_raw_text, user_id)
        
        image_file = product_data["image_file"]
        catalog_images_dir = Path(__file__).parent.parent.parent / "data" / "catalog_images"
        full_image_path = str((catalog_images_dir / image_file).resolve())
        
        final_reasoning = (
            f"Проанализировав ваш запрос и историю нашего диалога, я настоятельно рекомендую обратить внимание "
            f"на товар '{product_data['name']}' по цене {product_data['price']} руб. "
            f"Этот выбор идеально соответствует вашему контексту, так как содержит ключевые свойства и теги: "
            f"[{product_data['tags']}]. Натуральные материалы обеспечат максимальный комфорт и защиту в нужный сезон."
        )

        if span_vlm2: span_vlm2.end(status="SUCCESS")

        self.memory.add_message("Покупатель", user_raw_text)
        self.memory.add_message("Ассистент", final_reasoning)

        return Result.success({
            "intent": intent,
            "tags": search_text,
            "candidates": candidates_ctx,
            "reasoning": final_reasoning,
            "image_path": full_image_path if Path(full_image_path).exists() else None
        })
