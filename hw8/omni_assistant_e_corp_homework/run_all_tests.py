import sys
import unittest
from pathlib import Path

# Жестко фиксируем пути в системном рантайме ДО импорта тестов
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

# ЯВНЫЙ ИМПОРТ ТЕСТОВЫХ МОДУЛЕЙ (Полностью блокирует сброс контекста в /usr/lib/)
try:
    from tests.test_whisper_engine import TestWhisperASREngine
    from tests.test_omni_linguistic_and_denoise import TestOmniLinguisticAndDenoise
    print("✅ [LLMOPS] Все тестовые модули и зависимости (numpy, torch) успешно импортированы из ml_env!")
except ImportError as e:
    print(f"❌ Критическая ошибка импорта внутри ml_env: {e}")
    print("💡 Убедитесь, что запускаете скрипт через ./ml_env/bin/python3")
    sys.exit(1)

def main():
    print("🚀 [LLMOPS PROGRAMMATIC RUNNER v2] Ручная сборка тестового люкса в изолированном RAM-контуре...")
    
    # Явно собираем тестовый люкс (без вызова капризного loader.discover)
    suite = unittest.TestSuite()
    
    # Добавляем тест-кейсы из первого файла (test_whisper_engine.py)
    suite.addTest(unittest.makeSuite(TestWhisperASREngine))
    
    # Добавляем тест-кейсы из второго файла (test_omni_linguistic_and_denoise.py)
    suite.addTest(unittest.makeSuite(TestOmniLinguisticAndDenoise))

    print(f"📊 Всего загружено тест-кейсов для верификации: {suite.countTestCases()}")

    # Силовой запуск текстового раннера
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Если хотя бы один тест упал, возвращаем системный код ошибки для контроля сборки pack.sh
    if not result.wasSuccessful():
        sys.exit(1)
    else:
        print("\n✨ [FINAL TRIUMPH] Все мультимодальные модули ассистента E-Corp успешно верифицированы!")
        sys.exit(0)

if __name__ == "__main__":
    main()
