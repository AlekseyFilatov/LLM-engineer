import os
from huggingface_hub import hf_hub_download

# 1. Задаем базовую директорию вашего репозитория моделей Triton
BASE_DIR = os.path.expanduser("~/LLM-Training/src/triton_repository")
embedder_dir = os.path.join(BASE_DIR, "text_embedder/1")
classifier_dir = os.path.join(BASE_DIR, "text_classifier/1")
generator_dir = os.path.join(BASE_DIR, "text_generator/1")

# Убедимся, что целевые папки версий существуют
os.makedirs(embedder_dir, exist_ok=True)
os.makedirs(classifier_dir, exist_ok=True)
os.makedirs(generator_dir, exist_ok=True)

print("📥 1. Скачиваем ГОТОВЫЙ официальный ONNX-эмбеддер (bge-small-en-v1.5)...")
try:
    # Скачиваем напрямую скомпилированный ONNX файл из подпапки репозитория BAAI
    downloaded_embed = hf_hub_download(
        repo_id="BAAI/bge-small-en-v1.5",
        filename="onnx/model.onnx",
        local_dir=embedder_dir
    )
    # Переносим файл из подпапки onnx/ в корень папки версии 1/, как требует Triton
    target_embed_path = os.path.join(embedder_dir, "model.onnx")
    os.rename(os.path.join(embedder_dir, "onnx/model.onnx"), target_embed_path)
    
    # Очищаем пустую временную подпапку onnx/
    if os.path.exists(os.path.join(embedder_dir, "onnx")):
        os.rmdir(os.path.join(embedder_dir, "onnx"))
    print(f"✅ Эмбеддер успешно зафиксирован по пути: {target_embed_path}")
except Exception as e:
    print(f"❌ Ошибка скачивания эмбеддера: {e}")


# Путь к папке классификатора
classifier_dir = os.path.expanduser("~/LLM-Training/src/triton_repository/text_classifier/1")
os.makedirs(classifier_dir, exist_ok=True)

print("📥 2. Скачиваем ОФИЦИАЛЬНЫЙ и подтвержденный ONNX-классификатор (Xenova/bert-base-multilingual)...")
try:
    # Скачиваем нативный скомпилированный ONNX граф (он лежит в корне или подпапке onnx)
    downloaded_class = hf_hub_download(
        repo_id="Xenova/bert-base-multilingual-uncased-sentiment",
        filename="onnx/model.onnx",
        local_dir=classifier_dir
    )
    # Переносим из временной подпапки onnx/ строго в корень версии 1/
    target_class_path = os.path.join(classifier_dir, "model.onnx")
    if os.path.exists(os.path.join(classifier_dir, "onnx/model.onnx")):
        os.rename(os.path.join(classifier_dir, "onnx/model.onnx"), target_class_path)
        os.rmdir(os.path.join(classifier_dir, "onnx"))
        
    print(f"🎉 [SUCCESS] Настоящий классификатор зафиксирован по пути: {target_class_path}")
except Exception as e:
    print(f"❌ Ошибка скачивания: {e}")


# Путь к папке генератора текста в репозитории Triton
generator_dir = os.path.expanduser("~/LLM-Training/src/triton_repository/text_generator/1")
os.makedirs(generator_dir, exist_ok=True)

print("📥 3. Скачиваем РЕАЛЬНЫЙ ONNX-граф Qwen2.5-0.5B-Instruct (onnx-community)...")
try:
    # Скачиваем оптимизированную и сжатую (INT4) версию ONNX-декодера Qwen2.5
    downloaded_qwen = hf_hub_download(
        repo_id="onnx-community/Qwen2.5-0.5B-Instruct",
        filename="onnx/model_quantized.onnx",
        local_dir=generator_dir
    )
    
    target_qwen_path = os.path.join(generator_dir, "model.onnx")
    
    # Переносим из временной подпапки onnx/ строго в корень версии 1/ с переименованием
    if os.path.exists(os.path.join(generator_dir, "onnx/model_quantized.onnx")):
        os.rename(os.path.join(generator_dir, "onnx/model_quantized.onnx"), target_qwen_path)
        os.rmdir(os.path.join(generator_dir, "onnx"))
        
    print(f"🎉 [SUCCESS] Настоящая LLM Qwen2.5 зафиксирована по пути: {target_qwen_path}")
    
except Exception as e:
    print(f"❌ Ошибка скачивания Qwen2.5-0.5B: {e}")

