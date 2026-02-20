from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # API
    DEBUG: bool = False
    API_TITLE: str = "AI_picfilter API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://picfilter:password@localhost:5432/picfilter"
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # S3 / MinIO
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "picfilter-storage"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False

    # Image upload
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: list[str] = [
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    ]

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
