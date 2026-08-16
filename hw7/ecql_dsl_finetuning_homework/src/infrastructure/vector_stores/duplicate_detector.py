import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Final, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config.logger import logger
from config.settings import settings
from src.domain.monads import Result
from src.domain.interfaces import IVectorDuplicateDetector


class LocalANNFormatDetector(IVectorDuplicateDetector):
    """Высокопроизводительный двухфакторный детектор дубликатов корпоративного уровня.

    Защищен от дисковых блокировок WSL 2 и бесконечных циклов пагинации.
    """

    def __init__(self) -> None:
        self.db_path: Final[Path] = Path(settings.CHROMA_DB_DIR)
        self.collection_name: Final[str] = "dataset_generation_cache"
        
        logger.info("📡 [DETECTOR INIT] Запуск детектора дубликатов. Проверка конфигурации...")
        
        self.threshold_guard = settings.SIMILARITY_THRESHOLD
        if not (0.0 <= self.threshold_guard <= 1.0):
            logger.error(f"🚨 [DETECTOR] Ошибка конфигурации: порог {self.threshold_guard} вне диапазона. Откат к дефолту.")
            self.threshold_guard = settings.dup_detector_default_threshold

        device: Final[str] = settings.EMBEDDING_DEVICE if hasattr(settings, "EMBEDDING_DEVICE") else "cpu"
        
        # 1. Вычисляем точный абсолютный путь до папки src/ проекта
        # __file__ -> /mnt/d/LLM-Training/src/infrastructure/vector_stores/duplicate_detector.py
        current_file = Path(__file__).resolve()
        
        src_dir = current_file
        for parent in current_file.parents:
            if parent.name == "src":
                src_dir = parent
                break
                
        # Строим жесткий абсолютный путь: /mnt/d/LLM-Training/src/storage/embedding_model
        local_model_path = (src_dir / "storage" / "embedding_model").resolve()
        
        # Маркер готовности: папка существует и внутри лежит главный конфигурационный файл
        is_local_ready = local_model_path.exists() and (local_model_path / "config.json").exists()

        if is_local_ready:
            logger.info(f"🧠 [DETECTOR INIT] Найдена локальная копия. Загрузка С ДИСКА: {local_model_path}...")
            final_model_identifier = str(local_model_path)
            model_kwargs = {
                "device": device, 
                "local_files_only": True  # Намертво блокируем запросы в интернет
            }
        else:
            logger.warning(f"⚠️ [DETECTOR INIT] Локальная копия не найдена по пути: {local_model_path}")
            logger.info("📥 Первичный запуск: Скачивание модели 'sentence-transformers/all-MiniLM-L6-v2' из интернета...")
            
            # Создаем структуру папок, если её не было
            local_model_path.mkdir(parents=True, exist_ok=True)
            
            # Для скачивания используем строковое имя оригинального хаба
            final_model_identifier = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {"device": device}

        # 2. Инициализируем класс HuggingFaceEmbeddings
        self.embeddings: Final[HuggingFaceEmbeddings] = HuggingFaceEmbeddings(
            model_name=final_model_identifier,
            model_kwargs=model_kwargs
        )

        # 3. Если это был первый старт — атомарно запекаем веса на диск
        if not is_local_ready:
            logger.info(f"💾 Запись скачанных весов модели на диск: {local_model_path}...")
            
            if hasattr(self.embeddings, "_client"):
                self.embeddings._client.save_pretrained(str(local_model_path))
            elif hasattr(self.embeddings, "client"):
                self.embeddings.client.save_pretrained(str(local_model_path))
            else:
                logger.error("❌ Не удалось найти внутренний клиент sentence-transformers для сохранения весов.")
                
            logger.info("✅ Модель успешно закеширована локально! Интернет больше не потребуется.")

        
        logger.info(f"📂 [DETECTOR INIT] Подключение к постоянной базе ChromaDB по пути: {self.db_path}")
        chroma_settings = ChromaSettings(persist_directory=str(self.db_path), anonymized_telemetry=False)
        self.client: Final[chromadb.PersistentClient] = chromadb.PersistentClient(settings=chroma_settings)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.existing_ecql_codes: Set[str] = set()
        self.refresh_cache()

    def refresh_cache(self) -> None:
        """Метод инвалидации и полной перезагрузки кэша ОЗУ с безопасной пагинацией."""
        logger.info("🔄 [CACHE REFRESH] Сканирование базы данных и прогрев ОЗУ-кэша синтаксиса...")
        try:
            self.existing_ecql_codes.clear()
            
            # Безопасная проверка: если коллекция пуста, сразу выходим, не запуская циклы
            total_records = self.collection.count()
            if total_records == 0:
                logger.info("📊 [CACHE REFRESH] База данных пуста. Индексация истории не требуется.")
                return

            page_size = settings.dup_detector_page_size
            offset = 0
            
            # Жесткий лимит безопасности, чтобы цикл никогда не стал бесконечным
            max_safety_loops = (total_records // page_size) + 2
            loop_count = 0
            
            while offset < total_records and loop_count < max_safety_loops:
                loop_count += 1
                data = self.collection.get(
                    include=["metadatas"], 
                    limit=page_size, 
                    offset=offset
                )
                
                if not data or "metadatas" not in data or not data["metadatas"]:
                    break
                    
                for meta in data["metadatas"]:
                    if meta and "ecql_code" in meta:
                        self.existing_ecql_codes.add(self._normalize_code(meta["ecql_code"]))
                
                if len(data["metadatas"]) < page_size:
                    break
                offset += page_size
                
            logger.info(f"📊 [CACHE REFRESH] Успешно синхронизировано {len(self.existing_ecql_codes)} уникальных хэшей кода.")
        except Exception as e:
            logger.error(f"❌ [CACHE REFRESH] Критический крах пагинации: {str(e)}")
            self.existing_ecql_codes = set()

    @staticmethod
    def _normalize_code(code: str) -> str:
        return re.sub(r"\s+", " ", code.lower().strip())

    @staticmethod
    def _generate_id(text: str) -> str:
        return f"sha256_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def check_duplicates_and_update(self, new_pairs: List[Dict[str, str]], threshold: Optional[float] = None) -> Result[List[bool]]:
        """Двухфакторная проверка пачки кандидатов с безопасным маппингом индексов."""
        if not new_pairs:
            return Result.success([])
            
        current_threshold = threshold if threshold is not None else self.threshold_guard
        duplicate_mask: List[bool] = []
        new_items_to_add: List[Dict[str, Any]] = []
        
        try:
            # 1. Собираем только валидные тексты для пачечной векторизации
            valid_indices: List[int] = []
            input_texts: List[str] = []
            for idx, pair in enumerate(new_pairs):
                if pair and pair.get("input") and pair.get("output"):
                    valid_indices.append(idx)
                    input_texts.append(pair["input"])

            # Если ВСЯ пачка оказалась битой, мгновенно разворачиваем маску из True
            if not input_texts:
                return Result.success([True] * len(new_pairs))
                
            # Расчет эмбеддингов только для валидных строк
            new_vectors: List[List[float]] = self.embeddings.embed_documents(input_texts)
            initial_db_count = self.collection.count()
            
            # Маппим оригинальный индекс строки на её вектор
            vector_map = {valid_indices[i]: new_vectors[i] for i in range(len(valid_indices))}

            # 2. Поштучный последовательный анализ пачки
            for idx, pair in enumerate(new_pairs):
                # Защита от пустых или поврежденных словарей кандидатов
                if not pair or not pair.get("input") or not pair.get("output"):
                    duplicate_mask.append(True)
                    continue
                    
                input_clean: str = pair["input"].lower().strip()
                output_clean: str = self._normalize_code(pair["output"])
                vector: List[float] = vector_map[idx]
                
                # Фактор 1: Дедупликация по ОЗУ-кэшу хэшей ECQL
                if output_clean in self.existing_ecql_codes:
                    duplicate_mask.append(True)
                    continue
                    
                is_duplicate: bool = False
                
                # Фактор 2.1: Проверка на дубликаты внутри текущей пачки (In-Batch)
                if new_items_to_add:
                    for added_item in new_items_to_add:
                        cos_sim = sum(a*b for a, b in zip(vector, added_item["embedding"]))
                        if cos_sim > current_threshold:
                            is_duplicate = True
                            logger.debug(f"📡 [IN-BATCH FILTER] Внутренний дубликат пачки ({cos_sim:.2f} > {current_threshold})")
                            break
                            
                if is_duplicate:
                    duplicate_mask.append(True)
                    continue

                # Фактор 2.2: Запрос к векторному индексу ChromaDB на диске
                if initial_db_count > 0:
                    max_results = min(settings.dup_detector_n_results, initial_db_count)
                    query_res = self.collection.query(
                        query_embeddings=[vector], 
                        n_results=max_results, 
                        include=["distances"]
                    )
                    
                    if query_res and query_res.get("distances") and query_res["distances"] and query_res["distances"][0]:
                        for dist in query_res["distances"][0]:
                            similarity: float = 1.0 - float(dist)
                            if similarity > current_threshold:
                                is_duplicate = True
                                logger.debug(f"📡 [ANN FILTER] Обнаружен дубликат в БД (Сходство {similarity:.2f} > {current_threshold})")
                                break
                                
                if is_duplicate:
                    duplicate_mask.append(True)
                else:
                    # Элемент успешно прошел все фильтры
                    duplicate_mask.append(False)
                    new_items_to_add.append({
                        "id": self._generate_id(input_clean), 
                        "embedding": vector,
                        "document": input_clean, 
                        "metadata": {"ecql_code": output_clean}
                    })
                    self.existing_ecql_codes.add(output_clean)
                    
            # 3. Пакетный коммит уникальных векторов на жесткий диск
            if new_items_to_add:
                self.collection.add(
                    ids=[item["id"] for item in new_items_to_add],
                    embeddings=[item["embedding"] for item in new_items_to_add],
                    documents=[item["document"] for item in new_items_to_add],
                    metadatas=[item["metadata"] for item in new_items_to_add]
                )
                logger.info(f"💾 [DB UPDATE] Пакетная транзакция успешна. На диск сохранено +{len(new_items_to_add)} уникальных векторов.")
                
            return Result.success(duplicate_mask)
            
        except Exception as e:
            logger.exception(f"❌ [ANN DETECTOR] Критический крах пайплайна дедупликации: {str(e)}")
            return Result.failure(f"Ошибка детектора дубликатов: {str(e)}", "ANN_DETECTOR_FAILED")
