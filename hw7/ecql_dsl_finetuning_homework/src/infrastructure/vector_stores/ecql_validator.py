import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Final
import yaml

from src.domain.monads import Result
from config.settings import settings
from config.logger import logger


import re
import yaml
import logging
from typing import Any, Dict, List, Optional, Final
from pathlib import Path

# Предполагаем, что класс Result и settings импортированы корректно
# from your_modules import Result, settings

logger = logging.getLogger(__name__)

import re

class ECQLAutoCorrector:
    """Промышленный ИИ-фильтр автоматического исправления галлюцинаций модели Qwen."""
    
    @classmethod
    def patch_query(cls, query: str) -> str:
        if not query:
            return query
            
        q = query.upper().strip()
        
        # === Этап 1: Исправление галлюцинаций полей (Словарь синонимов) ===
        # Маппинг галлюцинируемых моделью полей на реальные поля из вашей YAML схемы
        field_replacements = {
            r'@price\b': '@amount',
            r'@cost\b': '@amount',
            r'@age\b': '@experience',          # Если возраст, переводим в стаж
            r'@purchase_year\b': '@deadline',   # Год закупки склада -> дедлайн/год
            r'@duration\b': '@deadline',        # Длительность проекта -> дедлайн
            r'@position\b': '@status',          # Должность сотрудника -> статус
            r'@department\b': '@city',          # Департамент -> город/локация
            r'@region\b': '@city',              # Регион -> город
            r'@title\b': '@status',             # Название позиции -> статус
            r'@start_date\b': '@deadline',      # Дата начала -> дедлайн (год)
            r'LIKE\s+[\'"]%?([^\'"]+)%?[\'"]': r"IS '\1'" # Превращаем LIKE '%ноутбук%' в IS 'ноутбук'
        }
        
        for pattern, replacement in field_replacements.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
            
        # === Этап 2: Тотальная зачистка скобок вокруг предикатов ===
        # Убираем конструкции вида (@field IS 'value') -> @field IS 'value'
        # Защищает от того, что скобки сломают парсер predicates в валидаторе
        q = re.sub(r'\(\s*(@[a-z_]+\s+[A-Z_]+\s+[^&|)]+)\s*\)', r'\1', q)
        
        # === Этап 3: Всеядный парсер BETWEEN для дат, строк и чисел ===
        # Паттерн захватывает любые значения, включая даты в кавычках '2023-05-01'
        between_regex = r'(@[a-z_]+)\s+BETWEEN\s+(\'[^\']+\'|"[^"]+"|[^\s]+)\s+AND\s+(\'[^\']+\'|"[^"]+"|[^\s]+)'
        
        def replace_between(match):
            field = match.group(1)
            val1 = match.group(2).strip()
            val2 = match.group(3).strip()
            
            # Очищаем кавычки вокруг значений, так как ABOVE/BELOW требуют чистые литералы
            val1_clean = re.sub(r"['\"]", "", val1)
            val2_clean = re.sub(r"['\"]", "", val2)
            
            # Если ИИ скормил дату '2023-05-01', забираем только год, чтобы поле осталось числовым!
            if '-' in val1_clean: 
                val1_clean = val1_clean.split('-')[0] # '2023-05-01' -> '2023'
            if '-' in val2_clean: 
                val2_clean = val2_clean.split('-')[0]
                
            return f"{field} ABOVE {val1_clean} && {field} BELOW {val2_clean}"
            
        q = re.sub(between_regex, replace_between, q, flags=re.IGNORECASE)

        # === Этап 4: Лечение оборванных генераций (Фоллбэк) ===
        # Если модель выдала обрубок '@date BETWEEN 01.01' или '@experience BETWEEN 1' без AND
        # Просто превращаем это в безопасный оператор ABOVE, чтобы запрос не падал по синтаксису
        broken_between_regex = r'(@[a-z_]+)\s+BETWEEN\s+([^\s&|]+)(?!\s+AND)'
        q = re.sub(broken_between_regex, r'\1 ABOVE \2', q, flags=re.IGNORECASE)
        
        # Убираем оставшиеся одиночные висячие скобки, если ИИ сошел с ума
        q = q.replace("(", "").replace(")", "")

        return q



class ECQLValidator:
    """Промышленный детерминированный компилятор-валидатор семантики и синтаксиса ECQL."""
    ALLOWED_ENTITIES: Final[set] = {"EMPLOYEES", "PROJECTS", "INVENTORY", "DEALS"}
    _schema_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _get_schema(cls) -> Dict[str, Any]:
        """Ленивая загрузка корпоративного каталога данных с безопасным разрешением путей."""
        if cls._schema_cache is not None:
            return cls._schema_cache
        try:
            # Универсальный поиск пути через центральные настройки
            schema_path: Path = Path(settings.DATA_SCHEMA_PATH).resolve()
            if not schema_path.exists():
                logger.warning(f"⚠️ [COMPILER] Файл схемы данных не найден по пути: {schema_path}")
                cls._schema_cache = {}
                return cls._schema_cache
            with schema_path.open("r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f) or {}
                cls._schema_cache = data.get("entities", {})
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"❌ [COMPILER] Сбой чтения конфигурации схемы: {str(e)}")
            cls._schema_cache = {}
        return cls._schema_cache

    @classmethod
    def validate_syntax(cls, query: str) -> Result[str]:
        """Выполняет лексический анализ, токенизацию и семантическую проверку типов (Type Checking)."""
        if not query or not query.strip():
            return Result.failure("Крах компиляции: Передан пустой запрос", "DSL_COMPILER_EMPTY_QUERY")     
        
        q: str = query.upper().strip()
        schema: Dict[str, Any] = cls._get_schema()
        
        if not q.startswith("FETCH"):
            return Result.failure(
                f"Крах компиляции: Запрос должен начинаться с FETCH: '{query}'",
                "DSL_COMPILER_PREFIX_ERROR"
            )
            
        entity_match: Optional[re.Match] = re.search(r"\[([A-Z_]+)\]", q)
        if not entity_match:
            return Result.failure(
                f"Крах компиляции: Не найдена сущность [ENTITY] в квадратных скобках: '{query}'",
                "DSL_COMPILER_ENTITY_MISSING"
            )
            
        entity_name: str = entity_match.group(1)
        if entity_name not in cls.ALLOWED_ENTITIES:
            return Result.failure(
                f"Крах компиляции: Неизвестная корпоративная сущность [{entity_name}]: '{query}'",
                "DSL_COMPILER_INVALID_ENTITY"
            )
            
        clean_q_no_quotes = re.sub(r"'[^']*'|\"[^\"]*\"", "", q)
        forbidden_sql_keywords: Final[List[str]] = ["SELECT", "FROM", "AND", "OR"]
        for word in forbidden_sql_keywords:
            if re.search(rf"\b{word}\b", clean_q_no_quotes, flags=re.IGNORECASE):
                return Result.failure(
                    f"Крах компиляции: Обнаружен запрещённый SQL-текст '{word}': '{query}'",
                    "DSL_COMPILER_SQL_INJECTION"
                )
                
        forbidden_operators: Final[List[str]] = ["=", ">", "<", ">=", "<=", "!=", "LIKE", "IN"]
        for op in forbidden_operators:
            if op.isalpha():
                has_op = re.search(rf"\b{op}\b", clean_q_no_quotes, flags=re.IGNORECASE)
            else:
                has_op = op in clean_q_no_quotes
            if has_op:
                return Result.failure(
                    f"Крах компиляции: Запрещённый SQL-оператор '{op}'. Используйте IS, NOT, ABOVE или BELOW: '{query}'",
                    "DSL_COMPILER_FORBIDDEN_OPERATOR"
                )
                
        if "WHERE" in q:
            try:
                parts_after_where: str = q.split("WHERE")[1].strip()
                where_block: str = re.split(r"\bAS\b", parts_after_where, flags=re.IGNORECASE)[0].strip()                
                predicates: List[str] = re.split(r'&&|\|\|', where_block)           
                for pred in predicates:
                    pred = pred.strip().strip("()")
                    if not pred: 
                        continue                                     
                    
                    # ИСПРАВЛЕНИЕ: Шаблон изменен на [a-zA-Z_], чтобы принимать заглавные поля после .upper()
                    triple_match: Optional[re.Match] = re.match(r'(@[a-zA-Z_]+)\s+(IS\s+NOT|IS|NOT|ABOVE|BELOW)\s+(.+)', pred, flags=re.IGNORECASE)
                    if not triple_match:
                        return Result.failure(
                            f"Синтаксическая ошибка в предикате: '{pred}'. Ожидался формат '@field OP value'",
                            "DSL_COMPILER_SYNTAX_ERROR"
                        )                                  
                    
                    # ИСПРАВЛЕНИЕ: field_name переводится в нижний регистр для успешной сверки со схемой yaml
                    field_name: str = triple_match.group(1).strip().lower()
                    operator: str = triple_match.group(2).upper().strip()
                    raw_value: str = triple_match.group(3).strip().rstrip(")")
                    
                    entity_fields: Dict[str, str] = schema.get(entity_name, {}).get("fields", {})
                    if field_name not in entity_fields:
                        return Result.failure(
                            f"Семантическая ошибка: Поле '{field_name}' отсутствует в схеме сущности [{entity_name}]",
                            "DSL_COMPILER_UNKNOWN_FIELD"
                        )                   
                        
                    field_description: str = entity_fields[field_name].lower()
                    is_numeric_field: bool = "число" in field_description
                    is_string_field: bool = "строка" in field_description
                    
                    if is_string_field:
                        if operator in ["ABOVE", "BELOW"]:
                            return Result.failure(
                                f"Несоответствие типов: Математическое сравнение '{operator}' недопустимо для строки '{field_name}'",
                                "DSL_COMPILER_TYPE_MISMATCH"
                            )
                        if not (raw_value.startswith("'") and raw_value.endswith("'")) and not (raw_value.startswith('"') and raw_value.endswith('"')):
                            return Result.failure(
                                f"Синтаксическая ошибка: Строковое значение {raw_value} для поля '{field_name}' должно быть в кавычках",
                                "DSL_COMPILER_STRING_QUOTES_ERROR"
                            )                 
                    if is_numeric_field:
                        clean_val: str = re.sub(r"['\"]", "", raw_value)
                        is_valid_number: bool = clean_val.replace('.', '', 1).isdigit()
                        
                        if not is_valid_number:
                            return Result.failure(
                                f"Несоответствие типов: К числовому полю '{field_name}' применено нечисловое значение '{raw_value}'",
                                "DSL_COMPILER_TYPE_MISMATCH"
                            )
                        if operator in ["ABOVE", "BELOW"] and (raw_value.startswith("'") or raw_value.startswith('"')):
                            return Result.failure(
                                f"Семантическая ошибка: Операторы ABOVE/BELOW не должны содержать кавычек: '{raw_value}'",
                               "DSL_COMPILER_NUMERIC_QUOTES_ERROR"
                            )
            except Exception as e:
                logger.error(f"❌ [COMPILER] Критический сбой разбора блока WHERE: {str(e)}")
                return Result.failure(f"Критический сбой семантического парсера: {str(e)}", "DSL_COMPILER_INTERNAL_ERROR")        
        return Result.success(query)