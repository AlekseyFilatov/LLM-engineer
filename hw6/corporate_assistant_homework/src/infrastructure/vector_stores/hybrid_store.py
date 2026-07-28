import os
import torch
import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

from src.domain.interfaces import VectorStoreInterface
from src.domain.monads import Result
from config.settings import settings

class CorporateHybridSearchStore(VectorStoreInterface):
    """Промышленный поисковый движок: ChromaDB + BM25 + Cross-Encoder Реранкер с монадической обработкой."""
    
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        hf_token = settings.HF_TOKEN if settings.HF_TOKEN.strip() else None

        # Получаем путь к кэшу из переменной окружения
        print("[INFO] Инициализация локального Cross-Encoder реранкера...")
        
        # Получаем путь к кэшу
        cache_dir = os.getenv("HF_HOME", "./storage/models")
        
        # Строим путь к глубокой папке, куда transformers складывает реальные файлы
        # Обычно это папка snapshots внутри репозитория модели
        model_snapshots_path = os.path.join(cache_dir, "hub/models--cross-encoder--ms-marco-MiniLM-L-2-v2/snapshots")
        
        # Проверяем, есть ли хоть один успешный скачанный snapshot модели
        local_only = False
        if os.path.exists(model_snapshots_path) and os.listdir(model_snapshots_path):
            # Заглядываем в первый попавшийся snapshot
            snapshot_dir = os.path.join(model_snapshots_path, os.listdir(model_snapshots_path)[0])
            # Кэш считается живым, только если там есть файлы конфигурации И весов!
            if os.path.exists(os.path.join(snapshot_dir, "config.json")):
                local_only = True

        if local_only:
            print("🔌 [RERANKER] Локальный кэш реранкера успешно верифицирован. Загрузка Offline.")
        else:
            print("🌐 [RERANKER] Локальный кэш пуст или поврежден. Скачивание чистой модели из сети...")

        # Инициализация
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-2-v2", 
            max_length=512,
            cache_folder=cache_dir,
            cache_dir=cache_dir,
            token=hf_token if hf_token else None,
            local_files_only=local_only  # Сюда прилетит True только если кэш 100% целый
        )
        self.bm25 = None
        self.raw_chunks_cache = []

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> Result[str]:
        """Метод для сохранения нарезки документов из парсера в ChromaDB."""
        try:
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            # Добавляем данные в коллекцию ChromaDB
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            
            # Сбрасываем кэш BM25, чтобы при следующем поиске индекс перестроился
            self.bm25 = None
            self.raw_chunks_cache = []
            
            return Result.success(f"Успешно сохранено {len(chunks)} векторов в ChromaDB.")
        except Exception as e:
            return Result.failure(f"Ошибка сохранения чанков в ChromaDB: {str(e)}", "CHROMA_WRITE_ERROR")

    def _build_bm25_index(self):
        """Выкачивает данные из ChromaDB и строит частотный индекс BM25 в ОЗУ."""
        all_docs = self.collection.get()
        if all_docs and all_docs["documents"]:
            self.raw_chunks_cache = [
                {"text doc": doc, "metadata": meta}
                for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
            ]
            tokenized_corpus = [doc.lower().split(" ") for doc in all_docs["documents"]]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search_similar(self, query: str, k: int = 3) -> Result[List[Dict[str, Any]]]:
        """Двухэтапный гибридный поиск с RRF-склейкой и жестким ML-барьером реранкера."""
        try:
            if not self.bm25 or not self.raw_chunks_cache:
                self._build_bm25_index()
                if not self.bm25:
                    return Result.success([]) # Возвращаем пустой успешный контейнер, если база пуста

            # Этап 1, Шлюз А: Векторный поиск
            chroma_res = self.collection.query(query_texts=[query], n_results=10)
            v_docs = chroma_res["documents"][0] if chroma_res and chroma_res["documents"] else []
            v_metas = chroma_res["metadatas"][0] if chroma_res and chroma_res["metadatas"] else []

            # Этап 1, Шлюз Б: Поиск по ключевым словам (BM25)
            tokenized_query = query.lower().split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_query)

            candidates = []
            seen_texts = set()

            # Объединяем кандидатов
            for doc, meta in zip(v_docs, v_metas):
                if doc not in seen_texts:
                    candidates.append({"text": doc, "metadata": meta})
                    seen_texts.add(doc)

            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
            for idx in top_bm25_indices:
                chunk = self.raw_chunks_cache[idx]
                if chunk["text doc"] not in seen_texts:
                    candidates.append({"text": chunk["text doc"], "metadata": chunk["metadata"]})
                    seen_texts.add(chunk["text doc"])

            if not candidates:
                return Result.success([])

            # Этап 2: Реранкинг Кросс-Энкодером
            pairs = [[query, c["text"]] for c in candidates]
            with torch.no_grad():
                rerank_scores = self.reranker.predict(pairs)

            for c, score in zip(candidates, rerank_scores):
                c["rerank_score"] = float(score)

            # Сортируем по качеству
            candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

            # Отфильтровываем жесткий мусор (Защита от галлюцинаций / Провокаций)
            final_top_k = []
            for c in candidates[:k]:
                if c["rerank_score"] >= -4.0:
                    final_top_k.append(c)

            return Result.success(final_top_k)

        except Exception as e:
            # Любая непредвиденная ошибка (проблемы с памятью CPU, сбой тензоров) безопасно перехватывается
            return Result.failure(
                error_message=f"Критический сбой гибридного поиска: {str(e)}",
                code="RETRIEVAL_ENGINE_FAILED"
            )
