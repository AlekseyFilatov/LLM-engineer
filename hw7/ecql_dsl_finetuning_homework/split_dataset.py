import json
import re
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from sklearn.model_selection import train_test_split

# Импортируем ваш центральный синглтон настроек
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def clean_and_format_prompt(user_input: str, ecql_code: str) -> dict:
    """Форматирует сырые данные из БД в итоговый датасет под Chat Template."""
    return {
        "instruction": "Переведи запрос на ECQL",
        "input": user_input.strip(),
        "output": ecql_code.strip()
    }

def split_dataset_from_db():
    logger.info("📡 [DB EXTRACT] Запуск выгрузки датасета напрямую из ChromaDB...")
    
    # 1. Настраиваем и инициализируем постоянный клиент ChromaDB по путям из Settings
    db_path = Path(settings.CHROMA_DB_DIR).resolve()
    if not db_path.exists():
        logger.error(f"❌ Векторная база данных не найдена по пути: {db_path}")
        return

    chroma_settings = ChromaSettings(persist_directory=str(db_path), anonymized_telemetry=False)
    client = chromadb.PersistentClient(settings=chroma_settings)
    
    # Имя коллекции совпадает с вашим детектором дубликатов
    collection_name = "dataset_generation_cache"
    collection = client.get_collection(name=collection_name)
    
    total_records = collection.count()
    logger.info(f"📊 Обнаружено записей в векторном индексе ChromaDB: {total_records}")
    
    if total_records < 10:
        logger.error("❌ Недостаточно данных в базе для формирования выборок!")
        return

    # 2. Выгружаем данные из БД с использованием безопасной потоковой пагинации
    dataset = []
    page_size = settings.dup_detector_page_size
    offset = 0
    
    logger.info(f"⏳ Считывание документов и метаданных пачками по {page_size} элементов...")
    
    while offset < total_records:
        data = collection.get(
            include=["documents", "metadatas"], 
            limit=page_size, 
            offset=offset
        )
        
        if not data or "documents" not in data or not data["documents"]:
            break
            
        # Итерируемся по выгруженной пачке
        for doc, meta in zip(data["documents"], data["metadatas"]):
            if doc and meta and "ecql_code" in meta:
                # doc — это очищенный русский текст (input)
                # meta['ecql_code'] — это откомпилированный ECQL запрос (output)
                formatted_pair = clean_and_format_prompt(doc, meta["ecql_code"])
                dataset.append(formatted_pair)
                
        if len(data["documents"]) < page_size:
            break
        offset += page_size

    logger.info(f"✅ Успешно выгружено и верифицировано {len(dataset)} пар.")

    # 3. Разделяем выгруженный массив с помощью scikit-learn (80/20)
    logger.info("✂️ Разделение выборки на Train и Test...")
    train_data, test_data = train_test_split(
        dataset, 
        test_size=0.20, 
        random_state=42, 
        shuffle=True
    )

    logger.info(f"📈 Обучающая выборка (Train): {len(train_data)} строк (80%)")
    logger.info(f"📉 Тестовая выборка (Test): {len(test_data)} строк (20%)")

    # 4. Создаем директорию data/ и записываем готовые .jsonl файлы
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = data_dir / "ecql_train.jsonl"
    test_file = data_dir / "ecql_test.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"✨ [SUCCESS] Файл обучения сохранен: {train_file}")
    logger.info(f"✨ [SUCCESS] Файл тестирования сохранен: {test_file}")

if __name__ == "__main__":
    split_dataset_from_db()
