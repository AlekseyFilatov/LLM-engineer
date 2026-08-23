import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import gradio as gr
from pydantic import validate_call

# Импорт конфигурационного слоя и Сердца проекта
from config.settings import settings
from src.use_cases.run_omni_assistant import RunOmniAssistantUseCase

# Глобальная конфигурация промышленного логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("omni_assistant")

class OrchestratorFactory:
    """
    Промышленная фабрика для управления жизненным циклом тяжелых Omni-моделей.
    Реализует паттерн Кэшируемого Синглтона для предотвращения повторной аллокации VRAM.
    """
    _instance: Optional[RunOmniAssistantUseCase] = None

    @classmethod
    def get_singleton(cls) -> RunOmniAssistantUseCase:
        if cls._instance is None:
            # Инициализируем модели строго один раз при первом обращении
            cls._instance = RunOmniAssistantUseCase()
        return cls._instance

@validate_call
def _validate_file_path(path: Optional[Path], field_name: str) -> Path:
    """Строгая Fail-Fast валидация входных бинарных потоков."""
    if path is None:
        raise ValueError(f"{field_name} не предоставлен.")
    if not path.exists():
        raise FileNotFoundError(f"{field_name} по пути {path} не найден.")
    if not path.is_file():
        raise ValueError(f"{field_name} должен быть файлом, а не директорией.")
    return path

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

def web_ui_orchestrator(image: Optional[str], audio_query: Optional[str]) -> Tuple[str, Optional[str], str]:
    """
    Высокоскоростной диспетчер веб-интерфейса.
    Выполняет валидацию, безопасный вызов пайплайна и форматирование LLMOps телеметрии.
    """
    try:
        if image is None or audio_query is None:
            return (
                "⚠️ Ошибка: Пожалуйста, загрузите изображение и запишите/загрузите аудио-вопрос.",
                None,
                "НЕТ ДАННЫХ"
            )

        # Фаза 1: Валидация существования файлов
        img_path = _validate_file_path(Path(image), "Изображение")
        aud_path = _validate_file_path(Path(audio_query), "Аудио-файл")

        # Фаза 2: Извлечение синглтона оркестратора из кэша фабрики
        orchestrator = OrchestratorFactory.get_singleton()
        
        # Фаза 3: Безопасный инференс сквозного пайплайна
        result = orchestrator.execute(image_path=img_path, audio_query_path=aud_path)

        if result.is_failure():
            logger.error(f"Сбой Omni-пайплайна: код={result.error_code}, ошибка={result.error}")
            # ИСПРАВЛЕНО: Убран синтаксический разрыв строки
            return (
                f"❌ Сбой Omni-пайплайна [{result.error_code}]: {result.error}",
                None,
                "ОШИБКА ВЫЧИСЛЕНИЙ"
            )

        # Фаза 4: Успешный разбор отчета и сбор Latency профайлинга
        report: Dict[str, Any] = result.value
        metrics = report.get("latency_report_seconds", {})
        
        total_time = metrics.get("total_pipeline", 0.0)
        asr_time = metrics.get("asr", 0.0)
        vlm_time = metrics.get("vlm", 0.0)
        tts_time = metrics.get("tts", 0.0)

        latency_md = (
            f"⏱️ **Общее время обработки:** {total_time:.2f} сек\n\n"
            f"*   **Фаза 1 (ASR Whisper Уши):** {asr_time:.2f} сек\n"
            f"*   **Фаза 2 (VLM Qwen2-VL Глаза):** {vlm_time:.2f} сек (обработка патчей геометрии)\n"
            f"*   **Фаза 3 (TTS Silero v5 Голос):** {tts_time:.2f} сек (запекание wav-матрицы)"
        )

        text_response = report.get("vlm_response_text", "Нет текстового ответа.")
        audio_path = report.get("output_audio_path")
        
        # ИСПРАВЛЕНО: Безопасное приведение Path к str для стабильности UI (Gradio)
        audio_path_str = str(audio_path) if audio_path else None
        
        logger.info(f"Запрос успешно обработан. Общее время: {total_time:.2f} сек")
        return text_response, audio_path_str, latency_md

    except FileNotFoundError as e:
        logger.warning(str(e))
        return str(e), None, "ФАЙЛ НЕ НАЙДЕН"
    except ValueError as e:
        logger.warning(str(e))
        return str(e), None, "НЕВЕРНЫЕ ДАННЫЕ"
    except Exception as e:
        logger.exception("Непредвиденная критическая ошибка в web_ui_orchestrator")
        return f"💥 Критическая системная ошибка: {str(e)}", None, "ВНУТРЕННЯЯ ОШИБКА"

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="zinc"
    )
    
    with gr.Blocks(theme=theme, title="E-Corp Omni-Assistant") as demo:
        gr.Markdown(
            """
            # 🤖 Мультимодальный Голосовой Omni-Ассистент E-Corp
            ### Сквозная интеграция технологий ASR (Whisper) + VLM (Qwen2-VL) + TTS (Silero v5)
            Весь конвейер работает на 100% автономно на тензорных ядрах видеокарты ноутбука!
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Входные модальности (Данные)")
                input_image = gr.Image(
                    label="📸 Загрузка изображения (Image/Context)",
                    type="filepath",
                    sources=["upload", "webcam"],
                    elem_id="input_image"
                )
                input_audio = gr.Audio(
                    label="🎙️ Запись или загрузка аудио-вопроса (Audio Query)",
                    type="filepath",
                    sources=["microphone", "upload"],
                    elem_id="input_audio"
                )
                submit_btn = gr.Button("🔥 Запустить Omni-конвейер", variant="primary", elem_id="submit_btn")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Выходные модальности (Ответ ассистента)")
                output_text = gr.Textbox(
                    label="💬 Текстовая расшифровка ответа Qwen2-VL",
                    interactive=False,
                    lines=6,
                    elem_id="output_text"
                )
                output_audio = gr.Audio(
                    label="🔊 Аудио-плеер голосового ответа (Silero v5)",
                    interactive=False,
                    autoplay=False,
                    elem_id="output_audio"
                )
                gr.Markdown("### 📈 LLMOps Профайлинг производительности")
                output_metrics = gr.Markdown(
                    value="*Ожидание запуска конвейера. Метрики Latency появятся после инференса.*",
                    elem_id="output_metrics"
                )
                
        submit_btn.click(
            fn=web_ui_orchestrator,
            inputs=[input_image, input_audio],
            outputs=[output_text, output_audio, output_metrics],
            api_name="omni_inference"
        )
        
        # Динамический вывод пути из центральных настроек settings
        gr.Markdown(
            f"""
            ---
            *Защищенный периметр E-Corp AI Labs. Локальный кэш NoSQL (shelve) активен. Логирование ведется в .*
            """
        )
    return demo

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "False").lower() == "true"
    
    app = build_app()
    logger.info(f"Запуск приложения на {server_name}:{server_port}, share={share}")
    app.launch(server_name=server_name, server_port=server_port, share=share)
