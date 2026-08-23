import logging
from typing import Dict, Any, Literal, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from src.domain.interfaces import PipelineState, BatchGenerationResponse
from src.infrastructure.vector_stores.ecql_validator import ECQLAutoCorrector, ECQLValidator
from src.infrastructure.vector_stores.duplicate_detector import LocalANNFormatDetector
from src.domain.monads import Result
from config.settings import settings

logger = logging.getLogger(__name__)


import logging
from typing import Any, Dict, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

class DatasetGeneratorGraph:
    """Оркестрируемый граф LangGraph для итеративной генерации корпоративного датасета.
    Включает в себя инъекцию схемы, Compiler-in-the-Loop валидацию типов
    и логарифмический векторный поиск дубликатов.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        logger.info("🤖 [GRAPH INIT] Старт ИИ-маршрутизатора графа генерации...")
        # Инициализируем один раз стандартный OpenAI-совместимый клиент для Ollama
        self.llm = ChatOpenAI(
            base_url=config["vllm_base_url"],
            api_key="ollama",
            model_name=config["model_name"],
            
            # === КРИТИЧЕСКИЕ НАСТРОЙКИ СДЕРЖИВАНИЯ МОДЕЛИ ===
            temperature=0.2,        # Сбиваем креативность. Модель начнет строго копировать схему
            model_kwargs={
                "top_p": 0.5,       # Ограничиваем выборку только самыми релевантными токенами
                "seed": 42,          # Фиксируем генерацию для стабильности пачек
                # Штраф за частоту: предотвращает повторение одних и тех же фраз/полей в пачке
                "frequency_penalty": 0.5, 
                # Штраф за присутствие: заставляет модель переключаться на новые темы/сущности
                "presence_penalty": 0.4   
            }
        )
        # Подключаем схему структурированного вывода (Pydantic модель пачки ответов)
        self.structured_llm = self.llm.with_structured_output(BatchGenerationResponse)
        self.detector = LocalANNFormatDetector()
        
        # Сборка структуры графа
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(PipelineState)
        
        workflow.add_node("generate", self.generation_node)
        workflow.add_node("validate", self.validation_node)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "validate")
        
        workflow.add_conditional_edges(
            "validate",
            self.routing_edge,
            {
                "generate": "generate",
                "end": END
            }
        )
        return workflow.compile()

    async def generation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Узел генерации пачки кандидатов моделью-учителем с инъекцией корпоративной схемы."""
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return state
            
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.config["prompts"]["ecql_generation"]),
                ("human", "Сгенерируй новую уникальную пачку из 10 запросов. Строго соблюдай каталогизацию.\nНЕ повторяй следующие уже созданные запросы:\n{history}")
            ])
            
            chain = prompt_template | self.structured_llm
            
            # Вытаскиваем реальную историю сгенерированных текстов для модели-учителя
            seen_list = state.get("seen_texts", [])
            history_str = ", ".join(seen_list[-30:]) if seen_list else "Пока нет" # Берем последние 30 для экономии контекста ноутбука
            
            # Асинхронный инференс Ollama со Schema Grounding
            response: BatchGenerationResponse = await chain.ainvoke({
                "data_catalog": getattr(settings, "data_catalog_string", "Базовый каталог"),
                "history": history_str
            })
            
            candidates: List[Dict[str, str]] = [pair.model_dump(by_alias=True) for pair in response.pairs]
            logger.info("--- [Node: Generate] Итерация №%d. Сгенерировано кандидатов: %d ---", state.get('loop_count', 0) + 1, len(candidates))
            
            return {
                "current_candidates": candidates,
                "loop_count": state.get("loop_count", 0) + 1,
                "error_rail": Result.success(None)
            }
        except Exception as e:
            logger.exception("❌ [Node: Generate] Критический сбой инференса модели-учителя")
            return {"error_rail": Result.failure(f"Ошибка инференса модели-учителя: {str(e)}", "LLM_GENERATION_FAILED")}

    async def validation_node(self, state: PipelineState) -> Dict[str, Any]:
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return state
        
        # Безопасное извлечение списков из состояния (защита от KeyError)
        candidates: List[Dict[str, str]] = state.get("current_candidates", [])
        existing_pool: List[Dict[str, str]] = list(state.get("task_pool", []))
        updated_seen: List[str] = list(state.get("seen_texts", []))
        
        threshold: float = state["config"]["similarity_threshold"]
        valid_candidates: List[Dict[str, str]] = []
        
        # 1. Шаг Compiler-in-the-Loop: строгая проверка синтаксиса и типов данных
        for item in candidates:

            patched_ecql = ECQLAutoCorrector.patch_query(item["output"])
            
            res = ECQLValidator.validate_syntax(patched_ecql)
            is_ok = res.is_success() if callable(res.is_success) else res.is_success            
            if is_ok:
                item["output"] = patched_ecql
                valid_candidates.append(item)
            else:
                logger.debug("   [Compiler Guard] Отклонен брак: %s", res.error)
                
        if not valid_candidates:
            logger.warning("⚠️ [Node: Validate] Ни один кандидат из текущей пачки не прошёл компилятор типов.")
            return {"current_candidates": [], "error_rail": Result.success(None)}

        # 2. Вызов двухфакторного детектора дубликатов (ChromaDB ANN HNSW)
        detector_res = self.detector.check_duplicates_and_update(valid_candidates, threshold)
        
        if detector_res.is_failure():
            return {"error_rail": detector_res}
            
        duplicate_mask: List[bool] = detector_res.value
        added_count: int = 0
        
        # 3. Наполнение финального пула и обновление истории уникальных текстов
        for idx, is_duplicate in enumerate(duplicate_mask):
            if not is_duplicate:
                candidate = valid_candidates[idx]
                existing_pool.append(candidate)
                # Записываем нормализованный запрос в историю
                updated_seen.append(candidate["input"].lower().strip())
                added_count += 1
                
        logger.info("--- [Node: Validate] Успешно добавлено в пул: %d пар. Всего в пуле: %d ---", added_count, len(existing_pool))
        
        # возвращаем СЛОВАРЬ со всеми обновленными полями состояния
        return {
            "task_pool": existing_pool,
            "seen_texts": updated_seen, # Теперь стейт LangGraph корректно запишет историю!
            "current_candidates": [],
            "error_rail": Result.success(None)
        }

    def routing_edge(self, state: PipelineState) -> Literal["generate", "end"]:
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return "end"
        
        cfg = state["config"]
        current_size = len(state.get("task_pool", []))
        target_size = cfg["total_target_samples"]
        max_loops = cfg["max_iterations"]
        
        if current_size >= target_size:
            logger.info("====== 🏁 ГЕНЕРАЦИЯ ЗАВЕРШЕНА: Цель в %d достигнута (Всего: %d) ======", target_size, current_size)
            return "end"
        if state.get("loop_count", 0) >= max_loops:
            logger.warning("====== 🛑 ГЕНЕРАЦИЯ ОСТАНОВЛЕНА: Достигнут лимит итераций (%d). Собрано: %d ======", max_loops, current_size)
            return "end"
            
        return "generate"

