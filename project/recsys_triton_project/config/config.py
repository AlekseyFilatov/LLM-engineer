import os
from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Централизованные конфигурации проекта E-Corp (Strict Pydantic v2 Standard)"""
    
    # Сетевые адреса распределенного ИИ-контура
    triton_grpc_url: str = "127.0.0.1:8001"
    triton_model_name: str = "paraphrase_embedder"
    whisper_url: str = "http://localhost:8003"
    
    # Сетевые настройки gRPC векторной базы Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6334
    qdrant_collection: str = "catalog"
    
    # Настройки Gradio UI хоста
    gradio_server_name: str = "127.0.0.1"
    gradio_server_port: int = 7860
    
    # Авторизационные токены Hugging Face и LLMOps мониторинга
    hf_token: str = ""
    langfuse_host: str = "http://localhost:3000"
    
    # Локальные пути к изолированным токенизаторам хоста
    local_tokenizer_path: str
    local_reranker_tokenizer_path: str

    # ИСПРАВЛЕНО: Декларативная конфигурация Pydantic v2 (Никаких class Config!)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Защита от падений, если в .env есть лишние переменные
    )

# Создаем синглтон настроек для импорта во все сервисы проекта
settings = Settings()


'''
class AppSettings(BaseSettings):
    # Добавлен еще один dirname, так как файл лежит в config/config.py
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    hf_token: str = Field(..., alias="HF_TOKEN")

    triton_grpc_url: str = Field(..., alias="TRITON_GRPC_URL")
    triton_model_name: str = Field(..., alias="TRITON_MODEL_NAME")
    triton_vlm_name: str = Field(..., alias="TRITON_VLM_NAME") 

    qdrant_host: str = Field(..., alias="QDRANT_HOST")
    qdrant_port: int = Field(..., alias="QDRANT_PORT")
    qdrant_collection: str = Field(..., alias="QDRANT_COLLECTION")

    whisper_url: HttpUrl = Field(..., alias="WHISPER_URL")
    langfuse_host: HttpUrl = Field(..., alias="LANGFUSE_HOST")
    local_tokenizer_path: str = Field(..., alias="LOCAL_TOKENIZER_PATH")
    gradio_server_name: str = Field(..., alias="GRADIO_SERVER_NAME")
    gradio_server_port: int = Field(..., alias="GRADIO_SERVER_PORT")

    local_tokenizer_path: str  # Будет мапиться на paraphrase_tokenizer
    local_reranker_tokenizer_path: str  # Будет мапиться на reranker_tokenizer

    class Config:
        env_file = ".env"
        extra = "ignore"

    @field_validator("local_tokenizer_path")
    @classmethod
    def expand_and_verify_path(cls, v: str) -> str:
        expanded = os.path.expanduser(v)
        return expanded

settings = AppSettings()
'''

