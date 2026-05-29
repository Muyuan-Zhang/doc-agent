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

    # M2 Retrieval
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 10
    final_top_k: int = 5
    # M5 Memory
    memory_recent_max_turns: int = 20
    memory_recent_ttl_seconds: int = 86400
    memory_summary_threshold: int = 15
    memory_milvus_collection: str = "memory_vectors"

    # M3 RAG Cache
    cache_ttl_seconds: int = 3600
    cache_auto_approve_threshold: int = 3
    cache_rewrite_enabled: bool = True
    cache_max_pending_reviews: int = 100
    cache_api_key: str = ""  # required for approve/reject/delete; empty disables auth

    # Redis Streams MQ
    mq_stream_name: str = "doc-agent:tasks"
    mq_consumer_group: str = "doc-agent-workers"
    mq_consumer_name: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}"
    )

    # M6 Consistency
    consistency_consumer_group: str = "doc-agent-consistency"
    consistency_consumer_name: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}-consistency"
    )
    cache_rag_namespace: str = "rag"
    cache_invalidation_scan_batch: int = 100
    cache_invalidation_max_iterations: int = 1000
    # M4 Agent
    agent_max_retries: int = 3
    agent_job_ttl_seconds: int = 3600
    stream_heartbeat_interval: float = 15.0
    agent_rate_limit_rpm: int = 20


settings = Settings()
