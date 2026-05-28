import hashlib
import logging
import re
from typing import TYPE_CHECKING

from app.core.config import Settings, settings as _default_settings

if TYPE_CHECKING:
    from app.clients.llm import AbstractLLMClient

logger = logging.getLogger(__name__)

# Match word chars + CJK unified ideographs + Hiragana/Katakana
_KEEP_PATTERN = re.compile(r"[^\w\s一-鿿぀-ヿ㐀-䶿]")


class QueryRewriter:
    def __init__(self, llm: "AbstractLLMClient", cfg: Settings | None = None) -> None:
        self._llm = llm
        self._cfg = cfg or _default_settings

    def normalize(self, query: str) -> str:
        q = query.strip().lower()
        q = _KEEP_PATTERN.sub(" ", q)
        return " ".join(q.split())

    def hash_query(self, normalized: str) -> str:
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def rewrite(self, query: str) -> tuple[str, str]:
        """Return (normalized_query, query_hash).

        Uses LLM to canonicalise the query when cache_rewrite_enabled=True.
        Falls back to plain normalize() if LLM is unavailable or disabled.
        """
        normalized = self.normalize(query)
        if self._cfg.cache_rewrite_enabled:
            try:
                prompt = (
                    "Rewrite the following search query into a canonical, concise form. "
                    "Return only the rewritten query, no explanation.\n\n"
                    f"Query: {query}"
                )
                rewritten = await self._llm.complete(prompt, max_tokens=64)
                if rewritten:
                    normalized = self.normalize(rewritten) or normalized
            except Exception as exc:
                logger.warning("cache=rewrite_failed fallback=normalize error=%s", exc)
        return normalized, self.hash_query(normalized)
