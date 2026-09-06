
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import time
import traceback

URL = "localhost:8000"
MODEL_NAME = "text_pipeline_smart"

# Справочник классов для реальной мультиязычной модели Xenova/bert-base-multilingual-uncased-sentiment
# Модель возвращает 5 классов (соответствующих оценке тональности от 1 до 5 звезд)
SENTIMENT_LABELS = {
    0: "⭐️ (Ужасно / Очень негативно)",
    1: "⭐️⭐️ (Плохо / Негативно)",
    2: "⭐️⭐️⭐️ (Нейтрально)",
    3: "⭐️⭐️⭐️⭐️ (Хорошо / Позитивно)",
    4: "⭐️⭐️⭐️⭐️⭐️ (Отлично / Очень позитивно)"
}

def softmax(x):
    """Стабильная функция Softmax для перевода логитов в вероятности"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

try:
    client = httpclient.InferenceServerClient(url=URL)
    
    if not client.is_server_live():
        print("❌ Ошибка: Triton Server не отвечает по адресу", URL)
        exit(1)
        
    if not client.is_model_ready(MODEL_NAME):
        print(f"❌ Ошибка: Модель '{MODEL_NAME}' не готова.")
        exit(1)
        
    print(f"✅ Успешное подключение к Triton Server!")
    print("💡 Введите 'exit' или 'выход' для завершения сессии.")
    print("="*60)

    # Запускаем интерактивный цикл общения с конвейером
    while True:
        prompt_text = input("\n📝 Введите текст для ИИ-конвейера: ").strip()
        
        if not prompt_text:
            continue
        if prompt_text.lower() in ['exit', 'выход', 'quit']:
            print("👋 Сессия успешно завершена. Удачи на защите!")
            break

        # --- ЭТАП 1: Подготовка и отправка данных ---
        prompt_data = np.array([prompt_text.encode('utf-8')], dtype=object)

        infer_input = httpclient.InferInput("prompt", prompt_data.shape, "BYTES")
        infer_input.set_data_from_numpy(prompt_data)
        inputs = [infer_input]

        outputs = [
            httpclient.InferRequestedOutput("generated_text"),
            httpclient.InferRequestedOutput("embeddings"),
            httpclient.InferRequestedOutput("logits")
        ]

        # Точный замер времени выполнения запроса со стороны клиента
        start_time = time.perf_counter()
        response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # --- ЭТАП 2: Разбор и декодирование текстового ответа ---
        gen_text_raw = response.as_numpy("generated_text")
        
        if gen_text_raw.dtype == object or gen_text_raw.dtype.kind in 'SU':
            generated_text = gen_text_raw.flatten()[0]
            if isinstance(generated_text, bytes):
                generated_text = generated_text.decode('utf-8')
        else:
            generated_text = bytes(gen_text_raw.flatten()[0]).decode('utf-8')

        # --- ЭТАП 3: Извлечение эмбеддингов и логитов ---
        embeddings = response.as_numpy("embeddings")
        logits = response.as_numpy("logits")

        # --- ЭТАП 4: Постобработка логитов BERT ---
        # Переводим сырые логиты батча в вероятности (0.0 - 1.0)
        probabilities = softmax(logits.flatten())
        predicted_class_idx = np.argmax(probabilities)
        predicted_label = SENTIMENT_LABELS.get(predicted_class_idx, "Неизвестный класс")
        confidence_percent = probabilities[predicted_class_idx] * 100

        # --- ЭТАП 5: Красивый вывод результатов в консоль ---
        print("\n" + "🚀 " + "="*25 + " РЕЗУЛЬТАТЫ BLS-ОКНА " + "="*25)
        print(f"⏱️  Время полного цикла (RTT Client-Side Latency): {latency_ms:.2f} мс")
        print("-" * 74)
        print(f"📝 1. Эхо-ответ сквозного промпта:\n   » {generated_text}")
        print("-" * 74)
        print(f"📊 2. Векторные эмбеддинги текста:")
        print(f"   • Геометрия тензора (Shape): {embeddings.shape} [батч, токенов, скрытая_размерность]")
        print(f"   • Первые 3 значения вектора: {embeddings.flatten()[:3]}")
        print("-" * 74)
        print(f"🔮 3. Интеллектуальный анализ тональности (BERT):")
        print(f"   • Предикт системы: {predicted_label}")
        print(f"   • Уверенность модели: {confidence_percent:.1f}%")
        print(f"   • Распределение вероятностей по всем 5 классам:")
        for idx, prob in enumerate(probabilities):
            print(f"     [Класс {idx+1}] {SENTIMENT_LABELS[idx][:15]}: {prob*100:.1f}%")
        print("🚀 " + "="*69)

except InferenceServerException as e:
    print("\n💥 [TRITON SERVER ERROR]:")
    print(f"Статус-код: {e.status()}")
    print(f"Сообщение от сервера:\n{e.message()}")

except Exception as e:
    print(f"\n❌ Локальная ошибка внутри client_test.py: {e}")
    print(f"Трейсбек локальной ошибки:\n{traceback.format_exc()}")
