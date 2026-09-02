import sys
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification 

class Colors:
GREEN = "\033[0;32m"
RED = "\033[0;31m"
RESET = "\033[0m" 

def log_step(msg: str, color: str = Colors.GREEN) -> None:
print(f"{color}📦 [CONVERTER]{Colors.RESET} {msg}") 

def ensure_dir(path: Path, description: str) -> None:
try:
path.mkdir(parents=True, exist_ok=True)
except OSError as e:
log_step(f"❌ Ошибка создания каталога {description}: {e}", Colors.RED)
sys.exit(1) 

def clean_extra_files(output_dir: Path) -> None:
"""
Удаляет конфигурационные json и текстовые файлы, созданные optimum,
оставляя только model.onnx. Это критично для бэкенда Triton!
"""
for file in output_dir.iterdir():
if file.is_file() and file.name != "model.onnx":
file.unlink() 

def download_and_export_embedder(
model_id: str,
output_dir: Path,
) -> None:
log_step(f"Загрузка и конвертация модели эмбеддингов: {model_id}...")
try:
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
model.save_pretrained(output_dir)
clean_extra_files(output_dir)
except Exception as e:
log_step(f"❌ Ошибка при конвертации эмбеддера: {e}", Colors.RED)
raise 

target = output_dir / "model.onnx"
if not target.exists():
log_step(f"❌ Файл model.onnx не создан в {output_dir}", Colors.RED)
sys.exit(1)
log_step("✅ Модель эмбеддингов успешно сохранена как model.onnx")
def download_and_export_classifier(
model_id: str,
output_dir: Path,
num_labels: int,
) -> None:
"""
Скачивает уже дообученный классификатор текста.
Параметр num_labels должен строго соответствовать оригинальной модели.
"""
log_step(f"Загрузка и конвертация готового классификатора: {model_id} (num_labels={num_labels})...")
try:
model = ORTModelForSequenceClassification.from_pretrained(
model_id,
export=True,
num_labels=num_labels,
)
model.save_pretrained(output_dir)
clean_extra_files(output_dir)
except Exception as e:
log_step(f"❌ Ошибка при конвертации классификатора: {e}", Colors.RED)
raise 

target = output_dir / "model.onnx"
if not target.exists():
log_step(f"❌ Файл model.onnx не создан в {output_dir}", Colors.RED)
sys.exit(1)
log_step("✅ Модель классификации успешно сохранена как model.onnx")
def prepare_generator_placeholder(output_dir: Path) -> None:
ensure_dir(output_dir, "text_generator/1")
marker = output_dir / "model.json"
marker.touch()
log_step("✅ Создан маркер model.json для text_generator (vLLM подтянет веса при старте)") 

def main() -> None:
project_root = Path.cwd()
triton_repo = project_root / "triton_repository" 

if not triton_repo.exists():
log_step(f"❌ Каталог {triton_repo} не найден. Запустите сначала setup_env.sh", Colors.RED)
sys.exit(1)
### 1. Эмбеддер

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_output_dir = triton_repo / "text_embedder" / "1"
ensure_dir(embed_output_dir, "text_embedder/1")
download_and_export_embedder(embed_model_id, embed_output_dir) 

### 2. Классификатор эмоций (6 классов)

classifier_model_id = "bhadresh-savani/bert-base-uncased-emotion"
classifier_num_labels = 6
classifier_output_dir = triton_repo / "text_classifier" / "1"
ensure_dir(classifier_output_dir, "text_classifier/1")
download_and_export_classifier(classifier_model_id, classifier_output_dir, classifier_num_labels) 

# 3. Генератор (заглушка для vLLM)

generator_ver_dir = triton_repo / "text_generator" / "1"
prepare_generator_placeholder(generator_ver_dir)

log_step("🎉 Все файлы моделей успешно сгенерированы и разложены по каталогам!")

if **name** == "**main**":
main()