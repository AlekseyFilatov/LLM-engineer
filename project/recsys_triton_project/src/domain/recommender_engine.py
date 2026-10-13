import re
from typing import Dict, Any, List
from config.logger import logger

class InMemoryRecommenderEngine:
    """
    Эмулятор промышленной рекомендательной системы (Stage 2 Personalization).
    Симулирует профили пользователей, матричное разложение ALS, ценовое сегментирование
    и контур политики Холодного Старта (Cold Start Policy) в оперативной памяти.
    """
    def __init__(self, boosting_registry: Dict[str, float], prompts_config: Dict[str, Any]):
        self.boosting_registry = boosting_registry
        self.default_margin = boosting_registry.get("дефолт", 1.0)
        # Читаем реестр пользователей напрямую из готового YAML-конфига prompts.yaml
        self.user_registry = prompts_config.get("user_registry", {})

        
        # 1. Справочник-эмулятор базы данных профилей пользователей (User Profile DB)
        self.user_registry = {
            "alexfil_premium": {
                "history": ["brown_suit.jpg", "blue_dress.jpg"], # Покупал костюм за 24k и платье за 12.5k
                "preferred_category": "одежда",
                "spending_segment": "premium",
                "is_new": False
            },
            "casual_buyer": {
                "history": ["white_shoes.jpg", "gray_sweater.jpg"], # Покупал кеды за 6k и свитер за 4k
                "preferred_category": "обувь",
                "spending_segment": "medium",
                "is_new": False
            },
            "new_user_cold_start": {
                "history": [], # Абсолютно пустая история, триггер Холодного Старта!
                "preferred_category": None,
                "spending_segment": "unknown",
                "is_new": True
            }
        }

    def compute_rec_features(self, user_id: str, item_name: str, item_price: float, item_file: str) -> Dict[str, float]:
        """
        Математический эмулятор генерации фичей CatBoost (Feature Engineering).
        Вычисляет ALS score, категориальное соответствие и коммерческий бустинг.
        """
        user_profile = self.user_registry.get(user_id, self.user_registry["casual_buyer"])
        
        # Динамически определяем категорию текущего товара на основе имени файла или тегов
        # (Одежда / Обувь / Аксессуары — берется из структуры вашего каталога)
        current_item_category = "аксессуары"
        if any(w in item_name.lower() for w in ["куртка", "платье", "свитер", "костюм", "дождевик"]):
            current_item_category = "одежда"
        elif any(w in item_name.lower() for w in ["кеды", "кроссовки", "обувь"]):
            current_item_category = "обувь"

        # --- КОНТУР 1: ПОЛИТИКА ХОЛОДНОГО СТАРТА (COLD START ROUTING) 🚨 ---
        if user_profile["is_new"]:
            # Для анонимного пользователя обнуляем персональный ALS-шум
            als_mock_score = 0.50
            
            # ИСПРАВЛЕНО: Вместо хардкода слов балансируем веса категорий верхнего уровня.
            # Базовый скор сегмента для холодного старта
            segment_match_score = 0.50
            
            # Извлекаем маржинальный буст из реестра маржинальности prompts.yaml
            m_weight = self.default_margin
            for keyword, weight in self.boosting_registry.items():
                if keyword != "дефолт" and keyword in item_name.lower():
                    m_weight = weight
                    break
            
            # ТЮНИНГ ВЕСОВ ДЛЯ ХОЛОДНОГО СТАРТА: 
            # Если товар имеет повышенный коммерческий бустинг (например, куртка с Boost: 1.3),
            # мы пропорционально подтягиваем segment_score, чтобы защитить контур от текстовых 
            # аномалий Тритона (вроде дождевика со Search: 0.94), сохраняя сезонный приоритет.
            if m_weight > 1.0:
                segment_match_score = 0.90
            elif m_weight < 1.0:
                segment_match_score = 0.20

            return {
                "als_score": als_mock_score,
                "segment_score": segment_match_score,
                "business_boost": m_weight,
                "is_cold_start": 1.0
            }

        # --- КОНТУР 2: ПРОГРЕТЫЙ ПОЛЬЗОВАТЕЛЬ (WARM USER) 🎯 ---
        # 1. Симуляция ALS на основе соответствия высокоуровневых категорий
        als_mock_score = 0.10
        if user_profile["preferred_category"] == current_item_category:
            als_mock_score = 0.85

        # 2. Симуляция ценового сегментирования (Spending Segment Match)
        segment_match_score = 0.50
        if user_profile["spending_segment"] == "premium":
            segment_match_score = 1.0 if item_price >= 10000 else 0.20
        elif user_profile["spending_segment"] == "medium":
            segment_match_score = 1.0 if 3000 <= item_price < 10000 else 0.30

        # 3. Базовый маржинальный буст
        m_weight = self.default_margin
        for keyword, weight in self.boosting_registry.items():
            if keyword != "дефолт" and keyword in item_name.lower():
                m_weight = weight
                break

        return {
            "als_score": als_mock_score,
            "segment_score": segment_match_score,
            "business_boost": m_weight,
            "is_cold_start": 0.0
        }

