import unittest
import numpy as np
from pathlib import Path

# Импортируем тестируемый инфраструктурный модуль и глобальные настройки
from src.infrastructure.asr.whisper_engine import WhisperASREngine
from config.settings import settings

class TestWhisperASREngine(unittest.TestCase):
    
    def setUp(self) -> None:
        """Инициализация тестового окружения перед каждым тестом."""
        self.max_duration = 10.0  # Жесткий лимит 10 секунд для быстрых тестов
        
        # Передаем тестовые параметры в конструктор движка
        self.engine = WhisperASREngine(
            model_name="base",
            max_duration_seconds=self.max_duration
        )

    def test_validate_audio_empty_array(self) -> None:
        """Проверка 1: Пустой массив амплитуд должен вызывать ValueError."""
        # Инициализируем пустой вектор амплитуд PCM
        empty_wave = np.array([], dtype=np.float32)
        
        with self.assertRaises(ValueError) as context:
            self.engine._validate_audio_length(
                empty_wave, 
                settings.TARGET_SAMPLE_RATE, 
                self.max_duration
            )
            
        # Проверяем на соответствие новой промышленной сигнатуре сообщения об ошибке
        self.assertIn("пуст или повреждён", str(context.exception))

    def test_validate_audio_too_long(self) -> None:
        """Проверка 2: Аудиозапись, превышающая max_duration, должна блокироваться."""
        # Вычисляем точное количество сэмплов для 11 секунд (на 1 секунду выше лимита)
        sample_rate = settings.TARGET_SAMPLE_RATE
        long_samples_count = sample_rate * 11
        
        # Генерируем тестовую синусоиду звуковой волны в памяти
        long_wave = np.sin(np.linspace(0, 440, long_samples_count)).astype(np.float32)
        
        with self.assertRaises(ValueError) as context:
            self.engine._validate_audio_length(
                long_wave, 
                sample_rate, 
                self.max_duration
            )
            
        self.assertIn("Аудио слишком длинное", str(context.exception))

    def test_denoise_no_silence_at_start(self) -> None:
        """
        Проверка 3: Защита от коротких записей.
        Если запись короче окна оценки шума (0.5 сек), спектральное вычитание
        должно вернуть исходный массив без математических крахов и делений на ноль.
        """
        sample_rate = settings.TARGET_SAMPLE_RATE
        # Создаем ультракороткий массив (всего 3 сэмпла), который заведомо меньше 0.5 секунд
        short_wave = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        
        # Принудительно отключаемnoisereduce для этого теста (если он установлен),
        # чтобы протестировать наш кастомный математический фоллбэк спектрального вычитания
        original_has_nr = getattr(WhisperASREngine, "_has_noisereduce", False)
        WhisperASREngine._has_noisereduce = False
        
        try:
            processed = self.engine._denoise_audio(short_wave, sample_rate)
            # Матрица не должна изменить свой размер или разрушиться
            self.assertEqual(len(processed), len(short_wave))
            np.testing.assert_array_almost_equal(processed, short_wave)
        finally:
            # Восстанавливаем исходный статус зависимости после теста
            WhisperASREngine._has_noisereduce = original_has_nr

    def test_normalize_audio_peak_scaling(self) -> None:
        """Проверка 4: Контур нормализации должен выравнивать пиковую амплитуду строго до 0.9."""
        # Создаем сигнал с избыточным пиком громкости (клиппинг на уровне 2.0)
        unnormalized_wave = np.array([-2.0, 0.0, 1.5, -0.5], dtype=np.float32)
        
        processed = self.engine._normalize_audio(unnormalized_wave)
        
        # Проверяем, что максимальный пиковый модуль равен ровно 0.9 (-1 dB)
        max_peak = np.max(np.abs(processed))
        self.assertAlmostEqual(max_peak, 0.9, places=5)

if __name__ == '__main__':
    unittest.main()
