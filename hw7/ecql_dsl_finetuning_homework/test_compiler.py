import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.infrastructure.vector_stores.ecql_validator import ECQLValidator

class TestECQLCompiler(unittest.TestCase):

    def test_correct_syntax(self):
        """Проверка 1: Идеальный валидный запрос должен проходить компиляцию."""
        query = "FETCH [EMPLOYEES] WHERE @city IS 'Moscow' && @salary ABOVE 150000 AS TABLE"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_success)

    def test_invalid_prefix(self):
        """Проверка 2: Запрос без FETCH на старте должен блокироваться."""
        query = "GET [EMPLOYEES] WHERE @city IS 'Moscow'"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_failure)
        self.assertEqual(res.error_code, "DSL_COMPILER_PREFIX_ERROR")

    def test_missing_entity(self):
        """Проверка 3: Отсутствие квадратных скобок сущности должно браковаться."""
        query = "FETCH EMPLOYEES WHERE @city IS 'Moscow'"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_failure)
        self.assertEqual(res.error_code, "DSL_COMPILER_ENTITY_MISSING")

    def test_unknown_field(self):
        """Проверка 4: Поля, отсутствующие в data_schema.yaml (Schema Grounding), должны вызывать семантический краш."""
        query = "FETCH [EMPLOYEES] WHERE @unexisting_field IS 'Value'"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_failure)
        self.assertEqual(res.error_code, "DSL_COMPILER_UNKNOWN_FIELD")

    def test_sql_injection_defense(self):
        """Проверка 5: Просачивание ключевых слов SQL (SELECT, FROM, OR) должно жестко пресекаться."""
        query = "FETCH [EMPLOYEES] WHERE @city IS 'Moscow' OR @salary > 100"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_failure)
        self.assertEqual(res.error_code, "DSL_COMPILER_SQL_INJECTION")

    def test_type_mismatch_string_math(self):
        """Проверка 6: Математическое сравнение ABOVE/BELOW для строковых полей (@city) должно выдавать ошибку типа."""
        query = "FETCH [EMPLOYEES] WHERE @city ABOVE 500"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_failure)
        self.assertEqual(res.error_code, "DSL_COMPILER_TYPE_MISMATCH")

    def test_string_quotes_error(self):
        """Проверка 7: Строковые значения без кавычек должны вызывать синтаксическую ошибку компилятора."""
        query = "FETCH [EMPLOYEES] WHERE @city IS Moscow"
        res = ECQLValidator.validate_syntax(query)
        self.assertTrue(res.is_failure)
        self.assertEqual(res.error_code, "DSL_COMPILER_STRING_QUOTES_ERROR")

if __name__ == "__main__":
    print("🧪 Запуск изолированного тестового стенда компилятора ECQL...")
    unittest.main()
