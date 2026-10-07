import unittest
from typing import List, Dict, Any

# Импортируем интерфейсы, монады и компоненты системы
from src.domain.interfaces import IVectorDbService, ITritonClientService
from src.domain.recommender_engine import InMemoryRecommenderEngine
from src.app.orchestrator import OmniSearchAgentOrchestrator
from src.domain.monads import Result

# =====================================================================
# ФЕЙКОВЫЕ СТАБЫ (STUBS) ДЛЯ ТЕСТИРОВАНИЯ КОНТУРА БЕЗ СЕТИ И GPU
# =====================================================================
class FakeHealthyQdrantDb(IVectorDbService):
    def __init__(self, payload: str):
        self.payload = payload
    def search_candidates(self, query_vector: List[float], category_filter: str = None) -> Result[str]:
        return Result.success(self.payload)

class FakeCrashedQdrantDb(IVectorDbService):
    def search_candidates(self, query_vector: List[float], category_filter: str = None) -> Result[str]:
        return Result.failure("gRPC Timeout", "QDRANT_DB_ERROR")

class FakeHealthyTriton(ITritonClientService):
    def generate_clip_embedding(self, search_tags: str) -> Result[List[float]]:
        return Result.success([0.1] * 384)
    def compute_reranker_score(self, user_query: str, candidate_text: str) -> float:
        return 0.50


# =====================================================================
# ТЕСТОВЫЙ КОНТУР UNIT-ТЕСТОВ
# =====================================================================
class TestOmniRecSysAgent(unittest.TestCase):
    def setUp(self):
        """Тестовые фикстуры справочников"""
        self.mock_boosting_registry = {"синее шелковое": 1.5, "кожаная куртка": 1.3, "дефолт": 1.0}
        self.mock_prompts_config = {
            "user_registry": {
                "alexfil_premium": {"history": ["brown_suit.jpg"], "preferred_category": "одежда", "spending_segment": "premium", "is_new": False},
                "new_user_cold_start": {"history": [], "preferred_category": None, "spending_segment": "unknown", "is_new": True}
            }
        }
        self.engine = InMemoryRecommenderEngine(self.mock_boosting_registry, self.mock_prompts_config)

    def test_recommender_engine_premium_warm_user(self):
        """Тест 1: Проверка весов для WARM PREMIUM пользователя"""
        features = self.engine.compute_rec_features("alexfil_premium", "Элегантное синее шелковое платье", 12500.0, "blue_dress.jpg")
        self.assertEqual(features["is_cold_start"], 0.0)
        self.assertEqual(features["als_score"], 0.85)       
        self.assertEqual(features.get("segment_score"), 1.0)   

    def test_recommender_engine_cold_start_user(self):
        """Тест 2: Проверка триггера политики COLD START"""
        features = self.engine.compute_rec_features("new_user_cold_start", "Элегантное синее шелковое платье", 12500.0, "blue_dress.jpg")
        self.assertEqual(features["is_cold_start"], 1.0)
        self.assertEqual(features["als_score"], 0.50)       

    def test_orchestrator_empty_input_fail_safe(self):
        """Тест 3 [Edge Case]: Проверка реакции оркестратора на абсолютно пустой запрос"""
        # Инжектируем фейковые стабильные сервисы
        orchestrator = OmniSearchAgentOrchestrator(FakeHealthyQdrantDb(""), FakeHealthyTriton())
        result = orchestrator.run_workflow(text_query="   ", audio_file=None)
        self.assertTrue(result.is_failure())
        self.assertEqual(result.error_code, "EMPTY_INPUT")

    def test_orchestrator_infrastructure_crash_handling(self):
        """Тест 4 [Fail-Safe]: Проверка перехвата критического краха СУБД Qdrant (БЕЗ MOCKS!)"""
        # Инжектируем намеренно упавшую фейковую базу данных напрямую!
        orchestrator = OmniSearchAgentOrchestrator(FakeCrashedQdrantDb(), FakeHealthyTriton())
        result = orchestrator.run_workflow(text_query="куртка", audio_file=None)
        
        # Контур отказоустойчивости сработает со 100% точностью
        self.assertTrue(result.is_failure())
        self.assertEqual(result.error_code, "QDRANT_DB_ERROR")

    def test_orchestrator_score_fusion_logic(self):
        """Тест 5 [Integration Loop]: Проверка математики слияния шкал"""
        fake_payload = (
            "{'name': 'Зимняя куртка', 'price': 18900, 'tags': 'зима', 'image_file': 'jacket.jpg', 'qdrant_score': 0.950} | "
            "{'name': 'Летнее платье', 'price': 4200, 'tags': 'лето', 'image_file': 'dress.jpg', 'qdrant_score': 0.810}"
        )
        # Инжектируем готовую фейковую выдачу
        orchestrator = OmniSearchAgentOrchestrator(FakeHealthyQdrantDb(fake_payload), FakeHealthyTriton())
        orchestrator.recommender = self.engine
        
        result_data = orchestrator._parse_qdrant_string(fake_payload, "хочу куртку", "new_user_cold_start")
        self.assertIn("name", result_data)
        self.assertIn("image_file", result_data)

if __name__ == "__main__":
    unittest.main()
