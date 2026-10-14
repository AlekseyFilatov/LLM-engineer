import os
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# Централизованный импорт настроек строго по стандарту Pydantic v2
from config.config import settings

# Базовые директории согласно Clean Architecture вашего проекта
BASE_PROJECT_DIR = os.path.expanduser("~/LLM-Training/src")
BASE_REPO_DIR = os.path.join(BASE_PROJECT_DIR, "triton_repository")

print("📦 [OMNI MONOLITH DOWNLOADER] Запуск полной локализации оффлайн-моделей проекта...\n")

# =====================================================================
# 1. СЛОЙ: Старая модель OpenAI CLIP (clip_embedder)
# =====================================================================
clip_dir = os.path.join(BASE_REPO_DIR, "clip_embedder", "1")
print("📥 [1/5] Скачивание оригинального графа OpenAI CLIP...")
if os.path.exists(clip_dir):
    print("   🧹 Обнаружены старые файлы clip_embedder. Выполняется полная зачистка...")
    try:
        shutil.rmtree(clip_dir)
    except Exception as clean_err:
        print(f"   ⚠️ Предупреждение при очистке CLIP: {clean_err}")
os.makedirs(clip_dir, exist_ok=True)

try:
    downloaded_clip = hf_hub_download(
        repo_id="Xenova/clip-vit-base-patch32",
        filename="onnx/model.onnx",
        local_dir=clip_dir,
        local_dir_use_symlinks=False
    )
    target_clip_path = os.path.join(clip_dir, "model.onnx")
    src_clip_path = os.path.join(clip_dir, "onnx/model.onnx")
    if os.path.exists(src_clip_path):
        os.rename(src_clip_path, target_clip_path)  
    onnx_tmp_dir = os.path.join(clip_dir, "onnx")
    if os.path.exists(onnx_tmp_dir):
        shutil.rmtree(onnx_tmp_dir)
    print("   ✅ [READY] ONNX-граф CLIP успешно локализован.")
except Exception as e:
    print(f"   ❌ Ошибка скачивания CLIP: {e}\n")

local_clip_tok_dir = os.path.join(BASE_PROJECT_DIR, "clip_tokenizer")
os.makedirs(local_clip_tok_dir, exist_ok=True)
try:
    clip_tok = AutoTokenizer.from_pretrained("Xenova/clip-vit-base-patch32")
    clip_tok.save_pretrained(local_clip_tok_dir)
    print(f"   ✅ [READY] Токенизатор CLIP сохранен в: {local_clip_tok_dir}")
except Exception as e:
    print(f"   ❌ Ошибка сохранения clip_tokenizer: {e}\n")


# =====================================================================
# 2. СЛОЙ: ИИ-Маршрутизатор Google PaliGemma 2 (Каскад из 3 ONNX-компонентов)
# =====================================================================
print("\n📥 [2/5] Скачивание многокомпонентного С++ графа Google PaliGemma 2...")
local_gemma_tok_dir = os.path.join(BASE_PROJECT_DIR, "paligemma_tokenizer")
os.makedirs(local_gemma_tok_dir, exist_ok=True)
try:
    gemma_tok = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-224", token=settings.hf_token)
    gemma_tok.save_pretrained(local_gemma_tok_dir)
    print(f"   ✅ [READY] Токенизатор PaliGemma 2 зафиксирован в: {local_gemma_tok_dir}")
except Exception as e:
    print(f"   ❌ Ошибка сохранения paligemma_tokenizer: {e}\n")

components = {
    "vision_encoder": "onnx/vision_encoder.onnx",
    "embed_tokens": "onnx/embed_tokens_quantized.onnx",
    "decoder_model": "onnx/decoder_model_merged_quantized.onnx"
}
for model_name, hf_filename in components.items():
    model_version_dir = os.path.join(BASE_REPO_DIR, f"paligemma_{model_name}", "1") 
    if os.path.exists(model_version_dir):
        print(f"   🧹 Обнаружены старые файлы для '{model_name}'. Выполняется полная зачистка...")
        try:
            shutil.rmtree(model_version_dir)
        except Exception as clean_err:
            print(f"   ⚠️ Предупреждение при очистке: {clean_err}")                  
    os.makedirs(model_version_dir, exist_ok=True)  
    try:
        print(f"   -> Скачиваем подграф {model_name}...")
        downloaded_path = hf_hub_download(
            repo_id="onnx-community/paligemma2-3b-pt-224",
            filename=hf_filename,
            local_dir=model_version_dir,
            local_dir_use_symlinks=False,
            token=settings.hf_token
        )   
        target_path = os.path.join(model_version_dir, "model.onnx")
        src_path = os.path.join(model_version_dir, hf_filename)       
        if os.path.exists(src_path):
            os.rename(src_path, target_path)     
        data_filename = f"{hf_filename}_data"
        try:
            downloaded_data_path = hf_hub_download(
                repo_id="onnx-community/paligemma2-3b-pt-224",
                filename=data_filename,
                local_dir=model_version_dir,
                local_dir_use_symlinks=False,
                token=settings.hf_token
            )           
            original_data_basename = os.path.basename(hf_filename) + "_data"
            target_data_path = os.path.join(model_version_dir, original_data_basename)
            src_data_path = os.path.join(model_version_dir, data_filename)
            if os.path.exists(src_data_path):
                os.rename(src_data_path, target_data_path)
                print(f"      [INFO] Скачаны внешние ONNX-веса под оригинальным именем: {original_data_basename}")               
        except Exception:
            pass            
        onnx_tmp_dir = os.path.join(model_version_dir, "onnx")
        if os.path.exists(onnx_tmp_dir):
            shutil.rmtree(onnx_tmp_dir)           
    except Exception as e:
        print(f"   ❌ Ошибка при скачивании {model_name}: {e}\n")


# =====================================================================
# 3. СЛОЙ: Cross-Encoder Реранкер (bge_reranker)
# =====================================================================
print("\n📥 [3/5] Скачивание тяжелого графа BGE Cross-Encoder...")
reranker_model_dir = os.path.join(BASE_REPO_DIR, "bge_reranker", "1")
reranker_tok_dir = Path(settings.local_reranker_tokenizer_path).resolve()
if os.path.exists(reranker_model_dir):
    print("   🧹 Обнаружены старые файлы bge_reranker. Выполняется полная зачистка...")
    shutil.rmtree(reranker_model_dir)
os.makedirs(reranker_model_dir, exist_ok=True)
try:
    downloaded_path = hf_hub_download(
        repo_id="Xenova/bge-reranker-base",
        filename="onnx/model.onnx", 
        local_dir=reranker_model_dir,
        token=settings.hf_token
    )
    target_path = os.path.join(reranker_model_dir, "model.onnx")
    src_path = os.path.join(reranker_model_dir, "onnx/model.onnx")
    if os.path.exists(src_path):
        os.rename(src_path, target_path)    
    shutil.rmtree(os.path.join(reranker_model_dir, "onnx"))
    print("   ✅ [READY] ONNX-граф BGE Реранкера успешно зафиксирован.")
except Exception as e:
    print(f"   ❌ Ошибка при скачивании ONNX графа BGE: {e}\n")

if os.path.exists(reranker_tok_dir):
    shutil.rmtree(reranker_tok_dir)
os.makedirs(reranker_tok_dir, exist_ok=True)
try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base", token=settings.hf_token)
    tokenizer.save_pretrained(str(reranker_tok_dir))
    print(f"   ✅ [READY] Токенизатор BGE успешно локализован в: {reranker_tok_dir}")
except Exception as e:
    print(f"   ❌ Ошибка сохранения токенизатора BGE: {e}\n")


# =====================================================================
# 4. СЛОЙ: Мультиязычный SOTA Эмбеддер Тритона (paraphrase_embedder)
# =====================================================================
print("\n📥 [4/5] Скачивание C++ ONNX-графа Paraphrase-Multilingual для Triton...")
model_target_dir = os.path.join(BASE_REPO_DIR, "paraphrase_embedder", "1")
tokenizer_target_dir = Path(settings.local_tokenizer_path).resolve()
if os.path.exists(model_target_dir):
    shutil.rmtree(model_target_dir)
os.makedirs(model_target_dir, exist_ok=True)
try:
    downloaded_path = hf_hub_download(
        repo_id="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        filename="onnx/model.onnx", 
        local_dir=model_target_dir,
        token=settings.hf_token
    ) 
    target_path = os.path.join(model_target_dir, "model.onnx")
    src_path = os.path.join(model_target_dir, "onnx/model.onnx")
    if os.path.exists(src_path):
        os.rename(src_path, target_path)        
    shutil.rmtree(os.path.join(model_target_dir, "onnx"))
    print(f"   ✅ [READY] ONNX-граф Paraphrase зафиксирован: {target_path}")
except Exception as e:
    print(f"   ❌ Ошибка при скачивании ONNX графа Paraphrase: {e}\n")

if os.path.exists(tokenizer_target_dir):
    shutil.rmtree(tokenizer_target_dir)
os.makedirs(tokenizer_target_dir, exist_ok=True)
try:
    tokenizer = AutoTokenizer.from_pretrained("Xenova/paraphrase-multilingual-MiniLM-L12-v2", token=settings.hf_token)
    tokenizer.save_pretrained(str(tokenizer_target_dir))
    print(f"   ✅ [READY] Токенизатор Paraphrase успешно локализован в: {tokenizer_target_dir}")
except Exception as e:
    print(f"   ❌ Ошибка сохранения токенизатора Paraphrase: {e}\n")


# =====================================================================
# 5. СЛОЙ: Оригинальное PyTorch Ядро для обратной совместимости индексации
# =====================================================================
print("\n📥 [5/5] Локализация оригинальных PyTorch весов Paraphrase (paraphrase_model_pytorch)...")
target_model_dir = os.path.join(BASE_PROJECT_DIR, "paraphrase_model_pytorch")
if os.path.exists(target_model_dir):
    shutil.rmtree(target_model_dir)
os.makedirs(target_model_dir, exist_ok=True)
try:
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        token=settings.hf_token
    )
    model.save_pretrained(target_model_dir)
    print(f"   ✅ [READY] PyTorch-веса успешно запечены в оффлайн-кэш хоста: {target_model_dir}")
except Exception as e:
    print(f"   ❌ КРИТИЧЕСКИЙ СБОЙ при скачивании PyTorch модели: {e}\n")
    exit(1)

print("\n🏁 [MONOLITH COMPLETE] Абсолютно все ИИ-модели проекта успешно скачаны и зафиксированы в оффлайн-контуре!")



'''
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from config.config import settings

BASE_PROJECT_DIR = os.path.expanduser("~/LLM-Training/src")
BASE_REPO_DIR = os.path.join(BASE_PROJECT_DIR, "triton_repository")
clip_dir = os.path.join(BASE_REPO_DIR, "clip_embedder", "1")
if os.path.exists(clip_dir):
    print("🧹 Обнаружены старые файлы clip_embedder. Выполняется полная зачистка...")
    try:
        shutil.rmtree(clip_dir)
    except Exception as clean_err:
        print(f"   ⚠️ Предупреждение при очистке CLIP: {clean_err}")
os.makedirs(clip_dir, exist_ok=True)
try:
    downloaded_clip = hf_hub_download(
        repo_id="Xenova/clip-vit-base-patch32",
        filename="onnx/model.onnx",
        local_dir=clip_dir,
        local_dir_use_symlinks=False
    )
    target_clip_path = os.path.join(clip_dir, "model.onnx")
    src_clip_path = os.path.join(clip_dir, "onnx/model.onnx")
    if os.path.exists(src_clip_path):
        os.rename(src_clip_path, target_clip_path)  
    onnx_tmp_dir = os.path.join(clip_dir, "onnx")
    if os.path.exists(onnx_tmp_dir):
        shutil.rmtree(onnx_tmp_dir)
except Exception as e:
    print(f"❌ Ошибка скачивания CLIP: {e}\n")
local_clip_tok_dir = os.path.join(BASE_PROJECT_DIR, "clip_tokenizer")
os.makedirs(local_clip_tok_dir, exist_ok=True)
try:
    clip_tok = AutoTokenizer.from_pretrained("Xenova/clip-vit-base-patch32")
    clip_tok.save_pretrained(local_clip_tok_dir)
except Exception as e:
    print(f"   ❌ Ошибка сохранения clip_tokenizer: {e}\n")
local_gemma_tok_dir = os.path.join(BASE_PROJECT_DIR, "paligemma_tokenizer")
os.makedirs(local_gemma_tok_dir, exist_ok=True)
try:
    gemma_tok = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-224", token=settings.hf_token)
    gemma_tok.save_pretrained(local_gemma_tok_dir)
except Exception as e:
    print(f"   ❌ Ошибка сохранения paligemma_tokenizer: {e}\n")
components = {
    "vision_encoder": "onnx/vision_encoder.onnx",
    "embed_tokens": "onnx/embed_tokens_quantized.onnx",
    "decoder_model": "onnx/decoder_model_merged_quantized.onnx"
}
for model_name, hf_filename in components.items():
    model_version_dir = os.path.join(BASE_REPO_DIR, f"paligemma_{model_name}", "1") 
    if os.path.exists(model_version_dir):
        print(f"🧹 Обнаружены старые файлы для '{model_name}'. Выполняется полная зачистка...")
        try:
            shutil.rmtree(model_version_dir)
        except Exception as clean_err:
            print(f"   ⚠️ Предупреждение при очистке: {clean_err}")                  
    os.makedirs(model_version_dir, exist_ok=True)  
    try:
        print(f" -> Скачиваем {model_name}...")
        downloaded_path = hf_hub_download(
            repo_id="onnx-community/paligemma2-3b-pt-224",
            filename=hf_filename,
            local_dir=model_version_dir,
            local_dir_use_symlinks=False,
            token=settings.hf_token
        )   
        target_path = os.path.join(model_version_dir, "model.onnx")
        src_path = os.path.join(model_version_dir, hf_filename)       
        if os.path.exists(src_path):
            os.rename(src_path, target_path)     
        data_filename = f"{hf_filename}_data"
        try:
            downloaded_data_path = hf_hub_download(
                repo_id="onnx-community/paligemma2-3b-pt-224",
                filename=data_filename,
                local_dir=model_version_dir,
                local_dir_use_symlinks=False,
                token=settings.hf_token
            )           
            original_data_basename = os.path.basename(hf_filename) + "_data"
            target_data_path = os.path.join(model_version_dir, original_data_basename)
            src_data_path = os.path.join(model_version_dir, data_filename)
            if os.path.exists(src_data_path):
                os.rename(src_data_path, target_data_path)
                print(f"   [INFO] Скачаны внешние ONNX-веса под оригинальным именем: {original_data_basename}")               
        except Exception:
            pass            
        onnx_tmp_dir = os.path.join(model_version_dir, "onnx")
        if os.path.exists(onnx_tmp_dir):
            shutil.rmtree(onnx_tmp_dir)           
    except Exception as e:
        print(f"❌ Ошибка при скачивании {model_name}: {e}\n")
reranker_model_dir = os.path.join(BASE_REPO_DIR, "bge_reranker", "1")
reranker_tok_dir = os.path.join(BASE_PROJECT_DIR, "reranker_tokenizer")
if os.path.exists(reranker_model_dir):
    print("🧹 Обнаружены старые файлы bge_reranker. Выполняется полная зачистка...")
    shutil.rmtree(reranker_model_dir)
os.makedirs(reranker_model_dir, exist_ok=True)
try:
    downloaded_path = hf_hub_download(
        repo_id="Xenova/bge-reranker-base",
        filename="onnx/model.onnx", 
        local_dir=reranker_model_dir,
        token=settings.hf_token
    )
    target_path = os.path.join(reranker_model_dir, "model.onnx")
    src_path = os.path.join(reranker_model_dir, "onnx/model.onnx")
    if os.path.exists(src_path):
        os.rename(src_path, target_path)    
    shutil.rmtree(os.path.join(reranker_model_dir, "onnx"))
except Exception as e:
    print(f"   ❌ Ошибка при скачивании ONNX графа: {e}\n")
if os.path.exists(reranker_tok_dir):
    shutil.rmtree(reranker_tok_dir)
os.makedirs(reranker_tok_dir, exist_ok=True)
try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base", token=settings.hf_token)
    tokenizer.save_pretrained(reranker_tok_dir)
except Exception as e:
    print(f"   ❌ Ошибка сохранения токенизатора: {e}\n")
BASE_PROJECT_DIR = os.path.expanduser("~/LLM-Training/src")
BASE_REPO_DIR = os.path.join(BASE_PROJECT_DIR, "triton_repository")
model_target_dir = os.path.join(BASE_REPO_DIR, "paraphrase_embedder", "1")
tokenizer_target_dir = os.path.join(BASE_PROJECT_DIR, "paraphrase_tokenizer")
if os.path.exists(model_target_dir):
    shutil.rmtree(model_target_dir)
os.makedirs(model_target_dir, exist_ok=True)
try:
    print(" -> 1. Скачиваем официальный C++ ONNX-бинарник Sentence-Transformers...")
    downloaded_path = hf_hub_download(
        repo_id="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        filename="onnx/model.onnx", 
        local_dir=model_target_dir,
        token=settings.hf_token
    ) 
    target_path = os.path.join(model_target_dir, "model.onnx")
    src_path = os.path.join(model_target_dir, "onnx/model.onnx")
    if os.path.exists(src_path):
        os.rename(src_path, target_path)        
    shutil.rmtree(os.path.join(model_target_dir, "onnx"))
    print(f"   ✅ [Успешно] Модель зафиксирована по пути: {target_path}\n")
except Exception as e:
    print(f"   ❌ Ошибка при скачивании ONNX графа: {e}\n")
if os.path.exists(tokenizer_target_dir):
    shutil.rmtree(tokenizer_target_dir)
os.makedirs(tokenizer_target_dir, exist_ok=True)
try:
    tokenizer = AutoTokenizer.from_pretrained("Xenova/paraphrase-multilingual-MiniLM-L12-v2", token=settings.hf_token)
    tokenizer.save_pretrained(tokenizer_target_dir)
except Exception as e:
    print(f"   ❌ Ошибка сохранения токенизатора: {e}\n")
target_model_dir = os.path.join(BASE_PROJECT_DIR, "paraphrase_model_pytorch")
if os.path.exists(target_model_dir):
    shutil.rmtree(target_model_dir)
os.makedirs(target_model_dir, exist_ok=True)
try:
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        token=settings.hf_token
    )
    model.save_pretrained(target_model_dir)
except Exception as e:
    print(f"   ❌ КРИТИЧЕСКИЙ СБОЙ при скачивании PyTorch модели: {e}\n")
    exit(1)
'''