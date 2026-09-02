import numpy as np
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL_NAME = "text_pipeline_smart"

# 1. Инициализация HTTP-клиента Triton
client = httpclient.InferenceServerClient(url=URL)

# 2. Подготовка входных данных (Промпт)
prompt_text = "Проверка работы сквозного BLS-конвейера."
prompt_data = np.array([prompt_text.encode('utf-8')], dtype=object)

infer_input = httpclient.InferInput("prompt", prompt_data.shape, "BYTES")
infer_input.set_data_from_numpy(prompt_data)

# 3. Выполнение одного запроса к BLS-дирижеру
response = client.infer(model_name=MODEL_NAME, inputs=[infer_input])

# 4. Извлечение и вывод результатов для верификации
generated_text = response.as_numpy("generated_text")[0].decode('utf-8')
embeddings = response.as_numpy("embeddings")
logits = response.as_numpy("logits")

print("✅ Пайплайн успешно отработал через один запрос!")
print(f" -> Возвращенный текст: {generated_text}")
print(f" -> Форма тензора эмбеддингов: {embeddings.shape}")
print(f" -> Форма тензора логитов классификатора: {logits.shape}")
