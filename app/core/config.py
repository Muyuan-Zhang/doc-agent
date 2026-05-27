import os
import socket

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSemaphoreLimits(BaseModel):
    interactive: int = 5
    background: int = 2
    audit: int = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # Application
    app_name: str = "doc-agent"
    debug: bool = False

    # PostgreSQL
    postgres_url: str = "postgresql+asyncpg://user:password@localhost:5432/docagent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_alias: str = "default"

    # Knowledge base — version prefix included in all cache keys
    knowledge_base_version: str = "v1"

    # LLM concurrency: three independent semaphore buckets
    llm_semaphore_limits: LLMSemaphoreLimits = LLMSemaphoreLimits()

    # Redis Streams MQ
    mq_stream_name: str = "doc-agent:tasks"
    mq_consumer_group: str = "doc-agent-workers"
    mq_consumer_name: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}"
    )


settings = Settings()
