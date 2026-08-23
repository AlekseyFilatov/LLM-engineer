import os
import sys
import time
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.hf_api import HfFolder

# Импортируем централизованный слой конфигурации проекта
from settings import settings

MAX_RETRIES = 3
RETRY_DELAY = 5

def ensure_dir(path: Path):
    """Гарантирует существование указанной директории на диске."""
    Path(path).mkdir(parents=True, exist_ok=True)

def is_vlm_downloaded(model_dir: Path) -> bool:
    """
    Валидация оффлайн-кэша VLM: Qwen требует наличия config.json 
    и хотя бы одного основного файла весов в формате .safetensors.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return False
    has_config = (model_path / "config.json").exists()
    has_weights = any(model_path.glob("*.safetensors")) or any(model_path.glob("model*.safetensors"))
    return has_config and has_weights

def is_asr_downloaded(model_dir: Path) -> bool:
    """
    Валидация оффлайн-кэша ASR: Нативный Whisper от OpenAI поставляется 
    в виде весов .pt (например, base.pt или model.fp16.pt).
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return False
    # Модель считается готовой, если найден хотя бы один файл .pt
    has_pt_weights = any(model_path.glob("*.pt"))
    return has_pt_weights

def download_with_retry(url: str, dest_path: Path, description: str) -> bool:
    """
    Скачивание тяжелых бинарных монолитов с повторными попытками.
    Защищено от кэш-ловушек: проверяет физический размер файла на диске.
    """
    dest_path = Path(dest_path)
    
    # ИСПРАВЛЕНИЕ КЭШ-БАГА SILERO: Файл считается валидным, только если он весит > 40 МБ
    if dest_path.exists() and dest_path.stat().st_size > 40 * 1024 * 1024:
        current_size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"ℹ️ {description} уже существует, верификация размера ({current_size_mb:.1f} MB) успешна.")
        return True
    elif dest_path.exists():
        # Если файл есть, но весит мало — это битый огрызок от обрыва сети, удаляем его
        current_size_kb = dest_path.stat().st_size / 1024
        print(f"⚠️ Обнаружен поврежденный файл {description} ({current_size_kb:.1f} KB). Удаляем и перекачиваем...")
        try:
            dest_path.unlink()
        except OSError as e:
            print(f"❌ Не удалось удалить поврежденный файл: {e}")
            return False

    # Контур скачивания с повторными попытками при сбоях связи
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"📥 {description} (попытка {attempt}/{MAX_RETRIES})...")
            urllib.request.urlretrieve(url, dest_path)
            
            # Контрольная проверка размера после скачивания
            if dest_path.exists() and dest_path.stat().st_size > 40 * 1024 * 1024:
                print(f"✅ {description} успешно скачан и верифицирован.")
                return True
            else:
                raise ValueError("Скачанный файл не прошел валидацию размера (слишком мал).")
                
        except Exception as e:
            print(f"❌ Ошибка при скачивании ({attempt}): {e}")
            if attempt < MAX_RETRIES:
                print(f"⏳ Повторная попытка через {RETRY_DELAY} сек...")
                time.sleep(RETRY_DELAY)
            else:
                print("❌ Все попытки исчерпаны. Проверьте интернет-соединение.")
                return False
    return False

def main():
    print("🚀 [DOWNLOAD] Запуск динамической оффлайн-загрузки мультимодального конвейера E-Corp...")
    
    # Принудительно изолируем системный кэш HF внутри папки проекта
    ensure_dir(settings.HF_HOME)
    os.environ["HF_HOME"] = str(settings.HF_HOME)
    
    # Безопасно извлекаем токен Hugging Face (если он заложен для приватных моделей)
    hf_token = getattr(settings, "HF_TOKEN", os.getenv("HF_TOKEN", None))
    if hf_token:
        HfFolder.save_token(hf_token)
        print("🔐 Токен Hugging Face успешно проинициализирован из настроек.")

    # =====================================================================
    # 1. СКАЧИВАНИЕ ГЛАЗ (VLM Qwen2-VL-2B-Instruct)
    # =====================================================================
    ensure_dir(settings.VLM_DIR)
    if is_vlm_downloaded(settings.VLM_DIR):
        print(f"ℹ️ [1/3] VLM модель уже полностью запечена на диск: {settings.VLM_DIR}")
    else:
        print(f"\n📥 [1/3] Скачивание VLM модели: {settings.VLM_REPO}...")
        try:
            snapshot_download(
                repo_id=settings.VLM_REPO,
                local_dir=settings.VLM_DIR,
                local_dir_use_symlinks=False,
                # Отсекаем дубли весов .bin/.pt, оставляя только чистый .safetensors
                ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5"],
                token=hf_token
            )
            print("✅ VLM модель успешно запечена на диск!")
        except Exception as e:
            print(f"❌ Сбой загрузки VLM: {e}")
            sys.exit(1)

    # =====================================================================
    # 2. СКАЧИВАНИЕ УШЕЙ (ASR Whisper Base)
    # =====================================================================
    ensure_dir(settings.ASR_DIR)
    if is_asr_downloaded(settings.ASR_DIR):
        print(f"ℹ️ [2/3] ASR модель Whisper уже полностью запечена на диск: {settings.ASR_DIR}")
    else:
        print(f"\n📥 [2/3] Скачивание ASR модели: {settings.ASR_REPO}...")
        try:
            snapshot_download(
                repo_id=settings.ASR_REPO,
                local_dir=settings.ASR_DIR,
                local_dir_use_symlinks=False,
                # ИСПРАВЛЕНИЕ: Убрали '*.pt' из игнорирования! Whisper ОБЯЗАН скачать оригинальный файл весов .pt
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
                token=hf_token
            )
            print("✅ ASR модель успешно запечена на диск!")
        except Exception as e:
            print(f"❌ Сбой загрузки ASR: {e}")
            sys.exit(1)

    # =====================================================================
    # 3. СКАЧИВАНИЕ ГОЛОСА (Silero TTS v5)
    # =====================================================================
    ensure_dir(settings.TTS_DIR)
    print(f"\n📥 [3/3] Настройка TTS монолита: {settings.TTS_URL}...")
    if not download_with_retry(settings.TTS_URL, settings.TTS_FILE, "Silero TTS v5 Монолит"):
        print("❌ Не удалось скачать оригинальный файл Silero TTS. Контур не завершен.")
        sys.exit(1)

    print("\n✨ [DOWNLOAD SUCCESS] Весь мультимодальный стек НАМЕРТВО запечен на Вашем Ext4 диске!")
    print("💡 Модели полностью переведены в автономный оффлайн-режим.")

if __name__ == "__main__":
    main()
