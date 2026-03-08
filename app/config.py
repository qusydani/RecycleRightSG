# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH: str = "models/best.pt"
    CONFIDENCE_THRESHOLD: float = 0.40
    MAX_UPLOAD_MB: int = 10
    ENVIRONMENT: str = "development"

    # Model metadata (used by /metrics and UI)
    MODEL_NAME: str = "YOLOv26n"
    MODEL_MAP50: float = 0.888
    MODEL_MAP50_95: float = 0.712
    MODEL_SIZE_MB: float = 5.4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
