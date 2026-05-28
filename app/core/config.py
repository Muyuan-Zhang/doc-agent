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
    postgres_url: str = "postgresql+asyncpg://localhost:5432/docagent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_alias: str = "default"

    # Knowledge base — version prefix included in all cache keys
    knowledge_base_version: str = "v1"

    # Document ingestion chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_dim: int = 1536
    embedding_batch_size: int = 100
    milvus_kb_collection: str = "knowledge_base"

    # LLM / Embedding
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # LLM concurrency: three independent semaphore buckets
    llm_semaphore_limits: LLMSemaphoreLimits = LLMSemaphoreLimits()

    # M5 Memory
    memory_recent_max_turns: int = 20
    memory_recent_ttl_seconds: int = 86400
    memory_summary_threshold: int = 15
    memory_milvus_collection: str = "memory_vectors"

    # M3 RAG Cache
    cache_ttl_seconds: int = 3600
    cache_auto_approve_threshold: int = 1
    cache_rewrite_enabled: bool = True
    cache_max_pending_reviews: int = 100

    # Redis Streams MQ
    mq_stream_name: str = "doc-agent:tasks"
    mq_consumer_group: str = "doc-agent-workers"
    mq_consumer_name: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}"
    )


settings = Settings()
