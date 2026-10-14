from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel

T = TypeVar('T')

class Result(BaseModel, Generic[T]):
    is_success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        return cls(is_success=True, value=value)

    @classmethod
    def failure(cls, error_message: str, code: str = "UNKNOWN_ERROR") -> 'Result[T]':
        # Подключаем логгер и автоматически пишем ошибку при ее создании
        from config.logger import logger
        logger.error(f"Код: {code} | Сообщение: {error_message}")
        return cls(is_success=False, error=error_message, error_code=code)


    def is_failure(self) -> bool:
        return not self.is_success
