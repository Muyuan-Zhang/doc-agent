from fastapi import APIRouter

router = APIRouter(prefix="/agent", tags=["agent"])

# LangGraph 节点注册顺序（M4 实现）:
# 1. query_rewrite      — 查询重写
# 2. retrieval          — 混合检索（调用 HybridRetriever）
# 3. entity_extraction  — pass-through（Graph RAG 预留）
# 4. rerank             — LLM 重排序
# 5. generate           — 流式输出
# 6. cache_write        — 写入 RAG 缓存
