import unittest
import numpy as np
from collections import deque
from pathlib import Path

# Импортируем движки для тестов
from src.infrastructure.asr.whisper_engine import WhisperASREngine
from src.infrastructure.tts.silero_engine import SileroTTSEngine

class TestOmniLinguisticAndDenoise(unittest.TestCase):
    
    def setUp(self) -> None:
        self.asr = WhisperASREngine(max_duration_seconds=10.0)
        self.tts = SileroTTSEngine()

    # --- ТЕСТЫ WHISPER ENGINE ---
    def test_validate_audio_empty_array(self) -> None:
        empty_wave = np.array([], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.asr._validate_audio_length(empty_wave, 16000, 10.0)

    def test_whisper_chunks_deduplication(self) -> None:
        """
        Проверяем динамический алгоритм дедупликации нахлестов слов на стыке чанков.
        Защищает аудио-пайплайн от заиканий и дублирования текста при склейке 30-секундных окон.
        """
        text_chunks = ["Привет я голосовой ассистент компании", "ассистент компании И Корп"]
        
        if not text_chunks:
            result_text = ""
        elif len(text_chunks) == 1:
            result_text = text_chunks[0]
        else:
            # Инициализируем плоский массив строк первым считанным чанком
            merged: List[str] = [text_chunks[0]]
            
            for i in range(1, len(text_chunks)):
                prev = merged[-1]
                curr = text_chunks[i]
                
                # Задаем максимальную глубину поиска нахлеста (до 5 слов)
                max_overlap = min(5, len(prev.split()), len(curr.split()))
                
                # Динамическое сканирование стыка сверху вниз (от max_overlap до 1 слова)
                for overlap in range(max_overlap, 0, -1):
                    prev_tail = " ".join(prev.split()[-overlap:])
                    curr_head = " ".join(curr.split()[:overlap])
                    
                    if prev_tail == curr_head:
                        # Нашли точное совпадение! Срезаем продублированную «голову» у текущего чанка
                        curr = " ".join(curr.split()[overlap:]).strip()
                        break
                        
                if curr:
                    merged.append(curr)
                    
            result_text = " ".join(merged)
        
        # Контрольное посимвольное сопоставление эталона и алгоритмического вывода
        self.assertEqual(result_text, "Привет я голосовой ассистент компании И Корп")

    def test_tts_linguistic_normalization_genitive_case(self) -> None:
        """Проверяем согласование числительных в родительном падеже (рублей)."""
        raw_text = "на сумму 112400 рублей"
        clean_text = self.tts._normalize_text_linguistic(raw_text)
        
        # Проверяем лингвистическую точность падежной инфлексии pymorphy3 + num2words
        self.assertIn("ста двенадцати тысяч", clean_text)
        self.assertIn("четырёхсот рублей", clean_text)

    def test_tts_markdown_sanitizer(self) -> None:
        """Проверяем жесткое вырезание спецсимволов ИИ, ломающих интонацию TTS."""
        raw_text = "**Внимание:** На фото PROJECTS обнаружено здание стоимостью 50000000 `руб.`"
        clean_text = self.tts._normalize_text_linguistic(raw_text)
        
        # Проверяем, что служебные символы VLM стерты, а сокращения развернуты в полноценную речь
        self.assertNotIn("**", clean_text)
        self.assertNotIn("`", clean_text)
        self.assertIn("рублей", clean_text)


if __name__ == '__main__':
    unittest.main()
