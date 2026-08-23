import re
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import re
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class ECQLEvaluator:
    """Промышленный калькулятор метрик для DSL-языка запросов (без LLM-судей).

    Реализует посимвольный Exact Match и токенный многомерный анализ структуры
    грамматики на базе частотности распределения токенов (Bag-of-Words).
    """

    @staticmethod
    def _extract_dsl_elements(query: str) -> List[str]:
        """Разбивает ECQL-запрос на атомарные синтаксические токены для матчинга."""
        if not query:
            return []
        
        # ИСПРАВЛЕНО: Добавлен флаг re.IGNORECASE, чтобы маска [a-z] ловила поля 
        # независимо от регистра строки (@salary и @SALARY), и экранированы спецсимволы связок
        token_pattern = r'(FETCH|\[[A-Z_]+\]|@[a-zA-Z_]+|\bIS\b|\bNOT\b|\bABOVE\b|\bBELOW\b|&&|\|\|)'
        tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)
        
        # Приводим токены к единому верхнему регистру для честного сравнения
        return [t.strip().upper() for t in tokens if t.strip()]

    @classmethod
    def compute_metrics(cls, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Вычисляет Exact Match, Precision, Recall и F1 на основе частотности токенов синтаксиса."""
        # Гвардейская защита от пустого ввода
        if not predictions or not references or len(predictions) != len(references):
            return {
                "exact_match_accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_micro": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0
            }

        # 1. Вычисляем Exact Match (Строгое посимвольное совпадение строк)
        clean_preds = [p.strip() for p in predictions]
        clean_refs = [r.strip() for r in references]
        exact_match_acc = accuracy_score(clean_refs, clean_preds)

        # 2. Потоковый токен-матчинг для Precision/Recall
        all_pred_tokens = []
        all_ref_tokens = []
        vocab = set()
        
        for p, r in zip(clean_preds, clean_refs):
            p_tokens = cls._extract_dsl_elements(p)
            r_tokens = cls._extract_dsl_elements(r)
            vocab.update(p_tokens + r_tokens)
            all_pred_tokens.append(p_tokens)
            all_ref_tokens.append(r_tokens)
            
        vocab_list = list(vocab)
        token_to_id = {t: i for i, t in enumerate(vocab_list)}

        y_pred_matrix = []
        y_true_matrix = []

        # ИСПРАВЛЕНО: Переход от бинарных флагов к матрице ЧАСТОТНОСТИ токенов (Bag-of-Words).
        # Теперь избыточная генерация или пропуск токенов моделью жестко штрафуют лосс и метрики!
        for p_toks, r_toks in zip(all_pred_tokens, all_ref_tokens):
            pred_vec = [0] * len(vocab_list)
            ref_vec = [0] * len(vocab_list)
            
            for t in p_toks:
                if t in token_to_id:
                    pred_vec[token_to_id[t]] += 1  # Накапливаем частоту вместо флага = 1
            for t in r_toks:
                if t in token_to_id:
                    ref_vec[token_to_id[t]] += 1   # Накапливаем частоту вместо флага = 1
                
            y_pred_matrix.append(pred_vec)
            y_true_matrix.append(ref_vec)

        # Вычисляем макро- и микро-метрики качества структуры DSL
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_matrix, y_pred_matrix, average="macro", zero_division=0
        )
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_matrix, y_pred_matrix, average="micro", zero_division=0
        )

        return {
            "exact_match_accuracy": float(exact_match_acc),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro)
        }
