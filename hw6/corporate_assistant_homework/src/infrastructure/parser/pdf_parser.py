import os
import re
import pdfplumber
from typing import List, Dict, Any, Generator


# Добавляем импорты из LangChain и модель эмбеддингов для анализа границ
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings

class AdvancedDocumentParser:
    """Промышленный генераторный парсер корпоративных PDF-структур."""
    def __init__(self):
        # Берем размеры чанков напрямую из нашего глобального config/settings.py
        # self.chunk_size = settings.CHUNK_SIZE
        # self.overlap = settings.CHUNK_OVERLAP
        print("[INFO] Инициализация семантического сплиттера предложений...")
        # Используем ту же модель эмбеддингов, что и для базы, чтобы оценивать схожесть предложений
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} # Для инференса сплиттера достаточно CPU
        )
        # Настраиваем сплиттер по порогу градиента (разрыву шаблона мысли)
        self.text_splitter = SemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile" # Разрыв по статистическому скачку дистанции
        )

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    def parse_directory_generator(self, data_dir: str) -> Generator[Dict[str, Any], None, None]:
        if not os.path.exists(data_dir):
            return
        counter = 0
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)

            if filename.lower().endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        
                        header_match = re.search(r"^[А-ЯЁа-яёa-zA-Z\s]{3,30}", text)
                        current_header = header_match.group(0).strip() if header_match else "Общие положения"

                        # ЗАМЕНА: Вместо старого тупого цикла while вызываем умный семантический сплиттер
                        # Он принимает текст страницы и возвращает список документов-предложений, объединенных темой
                        semantic_docs = self.text_splitter.create_documents([text])

                        for s_doc in semantic_docs:
                            chunk_text = s_doc.page_content
                            if len(chunk_text.strip()) > 20:
                                yield {
                                    "id": f"doc_chunk_{counter}",
                                    "text": chunk_text,
                                    "metadata": {
                                        "source": filename,
                                        "page": str(page_idx + 1),
                                        "header": current_header
                                    }
                                }
                                counter += 1
