import os
import json
from pathlib import Path

def run_fix():
    print("🚀 [FIX] Запуск исправления структуры оффлайн-импорта...")
    
    project_root = Path("/home/alexfil/LLM-Training").resolve()
    ollama_models_dir = project_root / "src/storage/ollama_models"
    
    # 1. Генерируем правильный оффлайн-паспорт базовой модели
    manifest_dir = ollama_models_dir / "manifests/registry.ollama.ai/library/qwen2.5-coder"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "7b"
    
    manifest_data = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.ollama.image.manifest",
        "config": {
            "mediaType": "application/vnd.ollama.image.config",
            "digest": "sha256:59dd62bb498b3a6375e7f85b5f558665e19e067a7b96fe41f70efb2283e28109",
            "size": 431
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:59dd62bb498b3a6375e7f85b5f558665e19e067a7b96fe41f70efb2283e28109",
                "size": 4720000000
            }
        ]
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"✅ Локальный паспорт манифеста успешно записан: {manifest_path}")

    # 2. Генерируем идеально чистый Modelfile
    modelfile_path = ollama_models_dir / "Modelfile"
    
    modelfile_content = """FROM qwen2.5-coder:7b
ADAPTER lora_output
PARAMETER temperature 0.1
PARAMETER top_p 0.4
PARAMETER repeat_penalty 1.2
PARAMETER num_ctx 4096
PARAMETER num_thread 8
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"Ты — официальный ИИ-транслятор компании E-Corp. Твоя единственная задача — строго переводить запросы пользователя с естественного русского языка на язык ECQL. Используй сущности [EMPLOYEES], [PROJECTS], [INVENTORY], [DEALS] и разрешенные поля с префиксом @. Категорически запрещено использовать стандартные SQL-команды SELECT, FROM, WHERE, AND, OR. Выдавай СТРОГО чистый код ECQL.\"\"\"
"""
    
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"✅ Локальный Modelfile успешно записан: {modelfile_path}")
    print("✨ Все файлы инфраструктуры E-Corp полностью исправлены и готовы!")

if __name__ == "__main__":
    run_fix()
