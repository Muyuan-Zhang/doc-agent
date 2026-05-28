from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ConversationTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    ts: float  # unix timestamp


class MemorySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary_id: str
    user_id: str
    session_id: str
    summary_text: str
    content_hash: str
    importance_score: float = 1.0
    structured_facts: dict[str, Any] = {}


class StaticFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    user_id: str
    content: str
    content_hash: str
    importance_weight: float = 1.0
    embedding: Optional[list[float]] = None


class MemoryContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    turns: list[ConversationTurn]
    summary: Optional[MemorySummary] = None
    static_facts: list[StaticFact]
