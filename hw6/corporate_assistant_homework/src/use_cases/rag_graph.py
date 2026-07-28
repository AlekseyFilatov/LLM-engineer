import copy
import re
from typing import List, Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from src.domain.interfaces import VectorStoreInterface, LLMInterface, CorporateAnswerSchema
from src.domain.monads import Result
from config.settings import settings


class SuperRAGState(TypedDict):
    question: str
    intent: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: CorporateAnswerSchema  # СТАЛО: теперь здесь Pydantic-объект!
    revision_count: int
    metrics: Dict[str, Any]
    status: str
    error_rail: Result

class CorporateRAGGraph:
    def __init__(self, vector_store: VectorStoreInterface, llm: LLMInterface):
        self.vector_store = vector_store
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(SuperRAGState)

        # Добавляем узлы
        workflow.add_node("agent_brain", self.agent_brain_node)
        workflow.add_node("search_layer", self.search_layer_node)
        workflow.add_node("generation_layer", self.generation_layer_node)
        workflow.add_node("reflection_layer", self.reflection_layer_node)

        workflow.add_edge(START, "agent_brain")

        # Настраиваем умные условные переходы (Railway Routing)
        workflow.add_conditional_edges(
            "agent_brain", 
            self.route_after_agent, 
            {"search_layer": "search_layer", "generation_layer": "generation_layer", END: END}
        )
        
        workflow.add_edge("search_layer", "generation_layer")
        workflow.add_edge("generation_layer", "reflection_layer")

        workflow.add_conditional_edges(
            "reflection_layer", 
            self.route_after_reflection, 
            {"generation_layer": "generation_layer", END: END}
        )

        return workflow.compile()

    # =====================================================================
    # АСИНХРОННЫЕ УЗЛЫ СУПЕР-ГРАФА
    # =====================================================================

    async def agent_brain_node(self, state: SuperRAGState) -> dict:
        print("\n🧠 [Monadic Graph] Шаг 1: ИИ-Агент анализирует интент запроса через LLM...")
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return state

        try:
            # 1. Извлекаем промпт маршрутизатора из YAML
            router_prompt = settings.prompts.get("agent_brain", {}).get("router_system_prompt", "")
            
            # 2. Формируем шаблон для Qwen
            messages = [
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": f"Вопрос сотрудника: {state['question']}"}
            ]
            
            # Применяем шаблон чата
            text_prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.llm.tokenizer([text_prompt], return_tensors="pt").to(self.llm.model.device)

            # Передаем инференс в поток, чтобы не блокировать асинхронное ядро
            def _sync_router():
                import torch, re
                from src.domain.interfaces import IntentSchema
                
                with torch.no_grad():
                    generated_ids = self.llm.model.generate(**inputs, generation_config=self.llm.gen_config)

                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                decoded_list = self.llm.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                raw_json = decoded_list[0].strip() if decoded_list else "{}"
                
                # Клининг возможных markdown-оберток модели
                raw_json = re.sub(r"^```json\s*|```$", "", raw_json, flags=re.MULTILINE).strip()
                
                # Ювелирно отрезаем лишние trailing characters, если они возникнут
                s_idx = raw_json.find('{')
                e_idx = raw_json.rfind('}')
                if s_idx != -1 and e_idx != -1:
                    raw_json = raw_json[s_idx:e_idx + 1]

                # Валидируем интент через Pydantic
                validated_intent = IntentSchema.model_validate_json(raw_json)
                return validated_intent.intent

            # Запускаем классификацию
            detected_intent = await asyncio.to_thread(_sync_router)
            print(f"🎯 [Brain Router] ИИ определил категорию запроса как: '{detected_intent}'")
            
            monadic_result = Result.success({"intent": detected_intent, "status": "Интенты распределены"})
            
        except Exception as e:
            # Если классификация сломалась (например, модель выдала битый JSON), 
            # по умолчанию включаем безопасный corporate режим, чтобы сработал поиск и Guardrails
            print(f"⚠️ Ошибка маршрутизации LLM: {str(e)}. Аварийный откат в режим 'corporate'")
            monadic_result = Result.success({"intent": "corporate", "status": "Откат в corporate"})

        return {
            "error_rail": Result.success(None),
            **(monadic_result.value if monadic_result.is_success else {"status": "FAILED"})
        }


    async def search_layer_node(self, state: SuperRAGState) -> dict:
        print("📡 [Monadic Graph] Шаг 2: Поиск знаний с монадическим перехватом ошибок...")
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return state

        res = self.vector_store.search_similar(state["question"], k=3)
        if res.is_failure():
            return {"error_rail": res, "status": "FAILED"}

        return {
            "error_rail": Result.success(None),
            "retrieved_docs": res.value, 
            "status": "Знания извлечены"
        }

    async def generation_layer_node(self, state: SuperRAGState) -> dict:
        print(f"🤖 [Monadic Graph] Шаг 3: Вызов ИИ-клиента (Ревизия №{state.get('revision_count', 0)})...")
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return state

        intent = state.get("intent", "general")
        docs = state.get("retrieved_docs", [])

        # --- НАДЕЖНЫЙ GUARD ОТ ПУСТОЙ БАЗЫ ДЛЯ КОРПОРАТИВНОГО ИНТЕНТА ---
        if intent == "corporate" and not docs:
            print("🛑 [Guard] База данных пуста! Блокировка инференса во избежание галлюцинаций.")
            
            # Импортируем нашу доменную схему ответа
            from src.domain.interfaces import CorporateAnswerSchema
            
            # Создаем РЕАЛЬНЫЙ Pydantic-объект, а не сырую строку!
            empty_answer = CorporateAnswerSchema(
                text_answer="Я не знаю. В корпоративной базе знаний нет документов, регламентирующих данный вопрос.",
                citations=[],
                confidence_score=0.0,
                needs_human_review=True
            )
            
            # Жесткий return! Мы мгновенно выходим из узла, возвращая Result.success с объектом
            return {
                "error_rail": Result.success(None),
                "final_answer": empty_answer, 
                "status": "Ответ сформирован (База пуста)"
            }
        # -----------------------------------------------------------------

        # Обычный сценарий, если документы НАЙДЕНЫ или это общий интент
        try:
            feedback = state.get("metrics", {}).get("critic_feedback", "")
            system_prompt = settings.general_prompt if intent == "general" else settings.corporate_prompt
            
            if feedback:
                system_prompt += settings.prompts["generation_layer"]["critic_alarm_prefix"].format(feedback=feedback)

            # Вызываем структурированную генерацию
            res = await self.llm.generate_structured_response_async(
                prompt=state["question"], 
                context=docs, 
                custom_system=system_prompt
            )
            
            if res.is_failure():
                return {"error_rail": res, "status": "FAILED"}

            new_rev = state.get("revision_count", 0) + (1 if feedback else 0)
            return {
                "error_rail": Result.success(None),
                "final_answer": res.value,  # Здесь лежит успешно отвалидированный Pydantic объект
                "revision_count": new_rev, 
                "status": "Ответ сгенерирован"
            }
        except Exception as e:
            return {
                "error_rail": Result.failure(f"LLM Generation Core Crash: {repr(e)}", "LLM_CRASH"), 
                "status": "FAILED"
            }


    async def reflection_layer_node(self, state: SuperRAGState) -> dict:
        print("🕵️ [Monadic Graph] Шаг 4: Проверка синтаксиса сносок через Pydantic структуру...")
        if state.get("error_rail") and state["error_rail"].is_failure(): 
            return state

        try:
            intent = state.get("intent", "general")
            pydantic_answer = state.get("final_answer")
            metrics = copy.deepcopy(state.get("metrics", {})) if state.get("metrics") else {}

            if intent == "general":
                metrics["is_valid"] = True
                return {"metrics": metrics, "error_rail": Result.success(None)}

            # --- УМНАЯ ПРОВЕРКА КРИТИКА НА ПУСТУЮ БАЗУ (ИСПРАВЛЕНИЕ ЗАЦИКЛИВАНИЯ) ---
            # Если уровень уверенности равен 0 или модель явно ответила, что не знает:
            if pydantic_answer and pydantic_answer.confidence_score == 0.0:
                print("🕵️ [Critic] Зафиксирован честный ответ об отсутствии данных. Проверка пройдена.")
                metrics["is_valid"] = True
                metrics["critic_feedback"] = ""
                return {"metrics": metrics, "error_rail": Result.success(None), "status": "Валидация завершена"}
            # ------------------------------------------------------------------------

            # Стандартная проверка для случаев, когда документы были найдены
            if not pydantic_answer or not pydantic_answer.citations:
                metrics["is_valid"] = False
                metrics["critic_feedback"] = settings.prompts["reflection_layer"]["missing_sources_error"]
            else:
                metrics["is_valid"] = True
                metrics["critic_feedback"] = ""

            return {"metrics": metrics, "error_rail": Result.success(None), "status": "Валидация завершена"}
        except Exception as e:
            return {"error_rail": Result.failure(f"Reflection Error: {str(e)}", "CRITIC_FAILED")}


    # =====================================================================
    # УМНЫЕ МОНАДИЧЕСКИЕ РОУТЕРЫ ГРАФА (Railway Routing)
    # =====================================================================

    def route_after_agent(self, state: SuperRAGState) -> str:
        if state.get("error_rail") and state["error_rail"].is_failure():
            print(f"🚨 [Монадический Роутер] Зафиксирован сбой: {state['error_rail'].error}. Аварийный выход.")
            return END

        if state.get("intent") == "corporate": 
            return "search_layer"
        return "generation_layer"

    def route_after_reflection(self, state: SuperRAGState) -> str:
        if state.get("error_rail") and state["error_rail"].is_failure():
            print(f"🚨 [Монадический Роутер] Крах на фазе рефлексии: {state['error_rail'].error}. Завершение графа.")
            return END

        if state.get("revision_count", 0) >= 3:
            print("🚨 [Loop Guard] Превышен лимит ревизий (3). Завершение графа.")
            return END

        if state.get("metrics", {}).get("is_valid") is False:
            print("🔄 [Reflection REJECT] Ссылки потеряны! Разворот графа на исправление...")
            return "generation_layer"

        print("✅ [Reflection APPROVE] Пайплайн успешно прошел все проверки.")
        return END

