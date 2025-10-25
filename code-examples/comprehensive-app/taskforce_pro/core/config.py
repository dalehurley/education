"""
Core configuration settings for TaskForce Pro.

Uses Pydantic Settings for environment variable validation.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Chapter 02: Pydantic models for configuration validation
    Chapter 11: Security settings for authentication
    """
    
    # Application
    APP_NAME: str = "TaskForce Pro"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = Field(..., min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database (Chapter 06)
    DATABASE_URL: str
    DATABASE_URL_SYNC: Optional[str] = None
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # Redis (Chapter 10)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    CACHE_DEFAULT_TTL: int = 300
    
    # Celery (Chapter 09)
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # AWS S3 / MinIO (Chapter 08)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: str = "taskforce-pro"
    AWS_REGION: str = "us-east-1"
    AWS_S3_ENDPOINT_URL: Optional[str] = None
    
    # File Upload (Chapter 04, 08)
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,gif,pdf,doc,docx,txt,csv,xlsx"
    
    # AI API Keys (Chapters 12-19)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-5"  # GPT-5 (best for coding and agentic tasks)
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-sonnet-4-5"  # Latest Claude Sonnet 4.5
    
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    # Vector Database (Chapter 14)
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "taskforce_documents"
    
    # Email (Chapter 09)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "noreply@taskforcepro.com"
    SMTP_FROM_NAME: str = "TaskForce Pro"
    
    # Rate Limiting (Production)
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # CORS (Chapter 03)
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Monitoring (Chapter 18)
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    
    # MLOps (Chapter 18)
    ENABLE_AB_TESTING: bool = True
    AB_TEST_SPLIT: float = 0.7
    MODEL_A: str = "gpt-5"  # GPT-5
    MODEL_B: str = "claude-sonnet-4-5"  # Claude Sonnet 4.5
    
    # Background Tasks (Chapter 09)
    ENABLE_TASK_REMINDERS: bool = True
    REMINDER_CHECK_INTERVAL: int = 3600  # 1 hour
    
    @field_validator("CORS_ORIGINS")
    @classmethod
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse comma-separated CORS origins into list."""
        return [origin.strip() for origin in v.split(",")]
    
    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def parse_allowed_extensions(cls, v: str) -> List[str]:
        """Parse comma-separated allowed extensions into list."""
        return [ext.strip() for ext in v.split(",")]
    
    @property
    def database_url_sync_resolved(self) -> str:
        """Get sync database URL, deriving from async URL if not set."""
        if self.DATABASE_URL_SYNC:
            return self.DATABASE_URL_SYNC
        # Convert asyncpg URL to psycopg2
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Singleton settings instance
settings = Settings()


# Laravel Comparison:
# This is similar to Laravel's config files and .env handling.
# - Pydantic Settings = Laravel's config/database.php, config/services.php
# - settings.DATABASE_URL = config('database.connections.pgsql.url')
# - Type validation = Laravel's config validation (less common)
# - Environment loading = .env file parsing with validation

