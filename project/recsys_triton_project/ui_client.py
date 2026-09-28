import time
import gradio as gr
from config.config import settings
from config.logger import logger
from langfuse import Langfuse
from pathlib import Path
from src.app.orchestrator import OmniSearchAgentOrchestrator
from src.infrastructure.qdrant_service import QdrantVectorSearchService
from src.infrastructure.triton_client import TritonInferenceClient

try:
    langfuse_client = Langfuse(host=str(settings.langfuse_host))
    logger.info("📊 [OMNI UI] Успешное подключение к Langfuse LLMOps.")
except Exception as lf_err:
    langfuse_client = None
    logger.warning(f"⚠️ Ошибка инициализации Langfuse в UI: {lf_err}")

# Создаем конкретные реализации согласно настройкам settings
qdrant_infra = QdrantVectorSearchService(settings.qdrant_host, settings.qdrant_port, settings.qdrant_collection)
triton_infra = TritonInferenceClient(
    settings.triton_grpc_url, 
    settings.triton_model_name, 
    Path(settings.local_tokenizer_path).resolve()
)

# Внедряем зависимости в оркестратор (Dependency Injection)
ai_agent = OmniSearchAgentOrchestrator(qdrant_service=qdrant_infra, triton_service=triton_infra)

def process_omni_search(text_query, audio_file):
    """Точка входа Gradio UI: маппинг монады Result в текст и изображение"""
    start_time = time.perf_counter()
    result = ai_agent.run_workflow(text_query, audio_file, langfuse_client=langfuse_client)
    latency = (time.perf_counter() - start_time) * 1000
    
    if result.is_failure():
        logger.error(f"💥 Пайплайн рухнул с кодом: {result.error_code}")
        # Возвращаем текст ошибки и None вместо картинки
        return f"❌ Ошибка Пайплайна [{result.error_code}]:\n{result.error}", None
        
    logger.info(f"✅ Пайплайн успешно отработал за {latency:.2f} мс")
    data = result.value
    
    formatted_candidates = "\n".join([f"• {c}" for c in data["candidates"].split(" | ")])

    text_output = (
        f"⏱️ Время сквозного инференса (RTT Latency): {latency:.2f} мс\n\n"
        f"🤖 ФИНАЛЬНЫЙ ОТВЕТ ИИ-АГЕНТА E-CORP:\n{data['reasoning']}\n\n"
        f"{'='*60}\n"
        f"📊 ТЕХНИЧЕСКИЕ МЕТАДАННЫЕ ВЫПОЛНЕНИЯ ПАЙПЛАЙНА:\n"
        f"🎯 Вычисленное ИИ-Намерение (Intent): '{data['intent']}'\n"
        f"🔍 Обогащенное семантическое ядро (Теги для CLIP): '{data['tags']}'\n"
        f"📥 Извлеченные из Qdrant DB сырые кандидаты:\n{formatted_candidates}"
    )

    # ИСПРАВЛЕНО: Возвращаем КОРТЕЖ из текстового лога и пути к картинке товара!
    return text_output, data.get("image_path")


# =====================================================================
# ОБНОВЛЕННЫЙ ИНТЕРФЕЙС GRADIO (ДВУХКОЛОНОЧНЫЙ ВЫВОД)
# =====================================================================
with gr.Blocks(title="Omni-Search E-Corp") as demo:
    gr.Markdown("# 🛒 Мультимодальный Omni-Поисковик E-Corp (Declarative Neural-Symbolic Architecture)")
    
    with gr.Row():
        # Левая колонка — Входные интерфейсы
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Текстовый запрос покупателя", placeholder="Я ищу что-то теплое для холодной зимы...")
            audio_input = gr.Audio(label="Голосовой поиск (Whisper ASR)", type="filepath")
            search_btn = gr.Button("🔍 Запустить Сквозной Пайплайн", variant="primary")
        
        # Правая колонка — Выходные интерфейсы (Текст + Нативная Фотография товара!)
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Аналитика и ответ ИИ-Агента", interactive=False, lines=12)
            output_image = gr.Image(label="Рекомендуемый товар из каталога E-Corp", type="filepath", height=300)

    # Связываем кнопку с функцией, указывая два выхода [output_text, output_image]
    search_btn.click(
        fn=process_omni_search, 
        inputs=[text_input, audio_input], 
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    import os; os.system("kill -9 $(lsof -t -i:7860) 2>/dev/null || true")
    demo.launch(server_name=settings.gradio_server_name, server_port=settings.gradio_server_port)
