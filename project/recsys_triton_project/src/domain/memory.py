from typing import List, Dict

class DialogueMemoryManager:
    """Менеджер контекстной памяти диалога (Инкапсуляция истории реплик)"""
    def __init__(self, max_history: int = 10):
        self._history: List[Dict[str, str]] = []
        self._max_history = max_history

    def add_message(self, role: str, text: str) -> None:
        self._history.append({"role": role, "text": text})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2:]

    def get_context_string(self) -> str:
        if not self._history:
            return "История пуста (первый запрос)"
        return "\n".join([f"{msg['role']}: {msg['text']}" for msg in self._history])

    def clear(self) -> None:
        self._history.clear()
