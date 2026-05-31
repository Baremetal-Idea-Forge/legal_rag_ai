from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Task API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Typesense
    TYPESENSE_HOST: str = "localhost"
    TYPESENSE_PORT: int = 8108
    TYPESENSE_PROTOCOL: str = "http"
    TYPESENSE_API_KEY: str = "xyz"
    TYPESENSE_CONNECTION_TIMEOUT: int = 5

    # Collections
    # TYPESENSE_TASKS_COLLECTION kept for backward compatibility
    TYPESENSE_TASKS_COLLECTION: str = "tasks"
    TYPESENSE_PDF_CHUNKS_COLLECTION: str = "pdf_chunks"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
