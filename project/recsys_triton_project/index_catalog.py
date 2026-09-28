import os
import re
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

# 1. Подключаемся к локальной инфраструктуре по gRPC
qdrant_client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
triton_client = grpcclient.InferenceServerClient(url="127.0.0.1:8001", verbose=False)

# 2. Загружаем локальный токенизатор хоста строго из конфига (нужен для подготовки длин фраз)
TOKENIZER_DIR = Path(__file__).parent / "src" / "paraphrase_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)

# Наш эталонный каталог товаров E-Corp
catalog_data = [
    {"file": "blue_dress.jpg", "name": "Элегантное синее шелковое платье", "price": 12500, "tags": "платье, синий, шелк, evening", "category": "одежда"},
    {"file": "green_dress.jpg", "name": "Летнее зеленое хлопковое платье", "price": 4200, "tags": "платье, зеленый, хлопок, лето", "category": "одежда"},
    {"file": "black_jacket.jpg", "name": "Зимняя черная кожаная куртка с мехом", "price": 18900, "tags": "куртка, черный, кожа, зима, мех", "category": "одежда"},
    {"file": "yellow_raincoat.jpg", "name": "Спортивный желтый дождевик водонепроницаемый", "price": 5500, "tags": "куртка, желтый, спорт, дождевик", "category": "одежда"},
    {"file": "red_sneakers.jpg", "name": "Красные спортивные беговые кроссовки", "price": 7800, "tags": "обувь, кроссовки, красный, спорт", "category": "обувь"},
    {"file": "white_shoes.jpg", "name": "Классические белые кожаные кеды", "price": 6200, "tags": "обувь, кеды, белый, кожа", "category": "обувь"},
    {"file": "brown_suit.jpg", "name": "Деловой коричневый шерстяной костюм", "price": 24000, "tags": "костюм, коричневый, шерсть, офис", "category": "одежда"},
    {"file": "gray_sweater.jpg", "name": "Теплый серый свитер крупной вязки", "price": 4800, "tags": "свитер, серый, шерсть, тепло", "category": "одежда"},
    {"file": "pink_backpack.jpg", "name": "Розовый детский школьный рюкзак", "price": 3100, "tags": "сумка, рюкзак, розовый, дети", "category": "аксессуары"},
    {"file": "sunglasses.jpg", "name": "Солнцезащитные очки авиаторы черные", "price": 2500, "tags": "аксессуары, очки, черный, лето", "category": "аксессуары"}
]

COLLECTION_NAME = "catalog"

# Пересоздаем чистую коллекцию Qdrant с геометрией 384
if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "image_vector": VectorParams(size=384, distance=Distance.COSINE),
        "text_vector": VectorParams(size=384, distance=Distance.COSINE),
    }
)
print(f"✅ Пересоздана чистая gRPC коллекция Qdrant '{COLLECTION_NAME}' (384-dims)!")

points = []

print("\n🚀 Запуск удаленной gRPC векторизации каталога внутри Triton Server на GPU...")
for idx, item in enumerate(catalog_data):
    try:
        # ИСПРАВЛЕНО: Оборачиваем строку в список [item["tags"]], чтобы гарантировать форму батча (1, 128)!
        inputs_tokenized = tokenizer([item["tags"]], padding="max_length", max_length=128, truncation=True, return_tensors="np")
        
        input_ids = inputs_tokenized["input_ids"].astype(np.int64)
        attention_mask = inputs_tokenized["attention_mask"].astype(np.int64)
        token_type_ids = inputs_tokenized.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

        # Упаковываем 3 gRPC-тензора идеальной формы (1, 128)
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
            grpcclient.InferInput("token_type_ids", token_type_ids.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        inputs[2].set_data_from_numpy(token_type_ids)

        # Вызов Triton на GPU по сети
        resp = triton_client.infer(model_name="paraphrase_embedder", inputs=inputs)
        
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
            except:
                continue

        if raw_vector is None:
            first_output_name = resp.get_response().outputs.name
            np_data = resp.as_numpy(first_output_name)
            if len(np_data.shape) == 3:
                raw_vector = np.mean(np_data, axis=1).flatten().tolist()
            else:
                raw_vector = np_data.flatten().tolist()

        # Математическая L2-нормализация вектора
        np_vec = np.array(raw_vector)
        norm = np.linalg.norm(np_vec)
        if norm > 0:
            np_vec = np_vec / norm
        vector_384 = np_vec.tolist()[:384]

        # Создаем точку для Qdrant
        point = PointStruct(
            id=idx,
            vector={
                "image_vector": vector_384,
                "text_vector": vector_384
            },
            payload={
                "name": item["name"],
                "price": item["price"],
                "tags": item["tags"],
                "image_file": item["file"],
                "category": item["category"]
            }
        )
        points.append(point)
        print(f" ➔ [gRPC TRITON ➔ QDRANT] Успешно залит товар: {item['name']}")

    except Exception as e:
        print(f" ❌ Ошибка векторизации товара '{item['name']}': {e}")

# Пушим пачку векторов в базу данных
if points:
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\n🎉 [SUCCESS] База данных Qdrant успешно наполнена {len(points)} товарами!")

'''
import os
import re
import yaml
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel

# Коннект по gRPC
client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)

print("📥 Загрузка локальной модели paraphrase-multilingual для переиндексации базы...")
# 1. Читаем токенизатор из локальной папки
tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / "src" / "paraphrase_tokenizer", local_files_only=True)

# 2. ИСПРАВЛЕНО: Читаем PyTorch веса СТРОГО из локальной папки без обращений к серверам Hugging Face!
LOCAL_PYTORCH_MODEL_DIR = Path(__file__).parent / "src" / "paraphrase_model_pytorch"
model = AutoModel.from_pretrained(str(LOCAL_PYTORCH_MODEL_DIR), local_files_only=True)
print("✅ Лингвистическое ядро успешно загружено в режиме OFFLINE.")

catalog_data = [
    {"file": "blue_dress.jpg", "name": "Элегантное синее шелковое платье", "price": 12500, "tags": "платье, синий, шелк, вечернее", "category": "одежда"},
    {"file": "green_dress.jpg", "name": "Летнее зеленое хлопковое платье", "price": 4200, "tags": "платье, зеленый, хлопок, лето", "category": "одежда"},
    {"file": "black_jacket.jpg", "name": "Зимняя черная кожаная куртка с мехом", "price": 18900, "tags": "куртка, черный, кожа, зима, мех", "category": "одежда"},
    {"file": "yellow_raincoat.jpg", "name": "Спортивный желтый дождевик водонепроницаемый", "price": 5500, "tags": "куртка, желтый, спорт, дождевик", "category": "одежда"},
    {"file": "red_sneakers.jpg", "name": "Красные спортивные беговые кроссовки", "price": 7800, "tags": "обувь, кроссовки, красный, спорт", "category": "обувь"},
    {"file": "white_shoes.jpg", "name": "Классические белые кожаные кеды", "price": 6200, "tags": "обувь, кеды, белый, кожа", "category": "обувь"},
    {"file": "brown_suit.jpg", "name": "Деловой коричневый шерстяной костюм", "price": 24000, "tags": "костюм, коричневый, шерсть, офис", "category": "одежда"},
    {"file": "gray_sweater.jpg", "name": "Теплый серый свитер крупной вязки", "price": 4800, "tags": "свитер, серый, шерсть, тепло", "category": "одежда"},
    {"file": "pink_backpack.jpg", "name": "Розовый детский школьный рюкзак", "price": 3100, "tags": "сумка, рюкзак, розовый, дети", "category": "аксессуары"},
    {"file": "sunglasses.jpg", "name": "Солнцезащитные очки авиаторы черные", "price": 2500, "tags": "аксессуары, очки, черный, лето", "category": "аксессуары"}
]

COLLECTION_NAME = "catalog"
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

# ИСПРАВЛЕНО: Размерность вектора теперь строго 384!
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "image_vector": VectorParams(size=384, distance=Distance.COSINE),
        "text_vector": VectorParams(size=384, distance=Distance.COSINE),
    }
)
print(f"✅ Создана чистая МНОГОВЕКТОРНАЯ коллекция '{COLLECTION_NAME}' (Размерность 384)!")

points = []
with torch.no_grad():
    for idx, item in enumerate(catalog_data):
        # Генерируем нативный чистый русский вектор из тегов
        inputs = tokenizer(item["tags"], padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        # Mean Pooling для извлечения вектора предложения размерности 384
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        vec = torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        
        # Нормализуем
        vec = vec / torch.norm(vec, p=2, dim=-1, keepdim=True)
        vector_384 = vec.numpy().flatten().tolist()

        point = PointStruct(
            id=idx,
            vector={
                "image_vector": vector_384,
                "text_vector": vector_384
            },
            payload={
                "name": item["name"],
                "price": item["price"],
                "tags": item["tags"],
                "image_file": item["file"],
                "category": item["category"]
            }
        )
        points.append(point)
        print(f"✅ Векторизован товар: {item['name']}")

if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\n🎉 База Qdrant успешно наполнена 384-мерными нативными русскими векторами!")
'''