# M0 实施计划：LangGraph 文档 Agent — 基础架构

> **状态：已完成** | 146 tests passed | 覆盖率 97%

---

## MQ 选型（Phase 3 新增）

当前栈已有 Redis，初版采用 Redis Streams，抽象层设计保留 M4 换后端的能力。

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| Redis Streams | 零新依赖、持久化、Consumer Group、背压可控 | 不适合高吞吐量（>10k msg/s） | **M0 初版采用** |
| RabbitMQ | 成熟、AMQP、死信队列 | 需新增基础设施 | M4 可迁移 |
| Kafka | 极高吞吐、回放能力 | 运维重，需 ZooKeeper/KRaft | 规模化后考虑 |

---

## 目录结构

```
doc-agent/
├── app/
│   ├── __init__.py            # create_app(), lifespan
│   ├── core/
│   │   ├── config.py          # Settings, LLMSemaphoreLimits
│   │   ├── exceptions.py      # 统一异常 + 注册函数
│   │   └── logging_config.py  # JSON 结构化日志
│   ├── models/
│   │   ├── chunk.py           # ChunkSchema（GraphRAG 预留）
│   │   └── retrieval.py       # RetrievalStrategy Protocol
│   ├── routers/
│   │   ├── health.py          # /health/live, /health/ready
│   │   └── agent.py           # M4 占位
│   ├── clients/
│   │   ├── base.py            # AbstractClient
│   │   ├── postgresql.py      # SQLAlchemy Core 连接池
│   │   ├── redis.py           # + increment_with_ttl, acquire_lock, release_lock
│   │   ├── milvus.py          # alias-only 访问
│   │   ├── mq.py              # AbstractMQClient + RedisStreamsMQClient
│   │   └── llm.py             # AbstractLLMClient（M4 实现，此处接口定义）
│   └── middleware/
│       ├── registry.py        # 统一中间件注册列表
│       └── request_id.py
├── tests/
│   ├── test_config.py
│   ├── test_lifespan.py
│   ├── test_health.py
│   ├── test_redis_client.py
│   ├── test_postgresql_client.py
│   ├── test_milvus_client.py
│   ├── test_mq_client.py
│   ├── test_models.py
│   └── test_exceptions.py
├── .env.example
├── pyproject.toml
└── docker-compose.yml
```

---

## Phase 1 — 配置

**文件：** `app/core/config.py`, `.env.example`, `pyproject.toml`

```python
class LLMSemaphoreLimits(BaseModel):
    interactive: int = 5
    background: int = 2
    audit: int = 1

class Settings(BaseSettings):
    app_name: str = "doc-agent"
    debug: bool = False

    # PostgreSQL（asyncpg 驱动）
    postgres_url: str = "postgresql+asyncpg://user:password@localhost:5432/docagent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_alias: str = "default"        # 唯一访问入口，不暴露 collection_name

    # 知识库版本（缓存 key 前缀）
    knowledge_base_version: str = "v1"

    # LLM 并发控制（三类独立信号量）
    llm_semaphore_limits: LLMSemaphoreLimits = LLMSemaphoreLimits()

    # Redis Streams MQ
    mq_stream_name: str = "doc-agent:tasks"
    mq_consumer_group: str = "doc-agent-workers"
    mq_consumer_name: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
```

---

## Phase 2 — 结构化日志 & 中间件

**文件：** `app/core/logging_config.py`, `app/middleware/request_id.py`, `app/middleware/registry.py`

### 日志字段（含 trace_id 预留）

```json
{
  "timestamp": "2026-05-27T10:00:00Z",
  "level": "INFO",
  "request_id": "uuid4",
  "trace_id": null,
  "message": "Request completed",
  "status_code": 200,
  "path": "/health",
  "service": "doc-agent"
}
```

`trace_id` 默认 `null`，接入 OpenTelemetry 时由 OTLP propagator 填充。`ContextVar` 同时持有 `request_id` 和 `trace_id`。

### 中间件注册（`middleware/registry.py`）

```python
# 注册顺序即执行顺序（外层先执行）
MIDDLEWARE_STACK = [
    # 插入点：AuthMiddleware（M4）
    # 插入点：RateLimitMiddleware（M4 流量控制）
    RequestIdMiddleware,     # 最内层，确保所有层都有 request_id
]

def register_middlewares(app: FastAPI) -> None:
    for mw in reversed(MIDDLEWARE_STACK):
        app.add_middleware(mw)
```

---

## Phase 3 — 客户端抽象层

### 基类（`app/clients/base.py`）

```python
class AbstractClient(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def ping(self) -> bool: ...
```

### RedisClient（`app/clients/redis.py`）

新增三个分布式工具方法：

```python
def cache_key(self, namespace: str, *parts: str) -> str:
    return f"{settings.knowledge_base_version}:{namespace}:{':'.join(parts)}"

async def increment_with_ttl(self, key: str, ttl_seconds: int, amount: int = 1) -> int:
    """原子 INCR + EXPIRE（Lua script），用于分布式限流计数器"""

async def acquire_lock(self, key: str, ttl_seconds: int, token: str | None = None) -> tuple[bool, str]:
    """SET key token NX EX ttl — 分布式锁，返回 (acquired, token)"""

async def release_lock(self, key: str, token: str) -> bool:
    """Lua script 验证 token 后 DEL，防止误释放他人锁"""
```

> 多 key 操作使用 hash tag（`{user:u123}:rate_limit`），避免 Cluster CROSSSLOT 错误。

### MilvusClient（`app/clients/milvus.py`）

强制 alias 访问，调用方不得传入原始 `collection_name`：

```python
class MilvusClient(AbstractClient):
    def __init__(self):
        self._alias = settings.milvus_alias   # 唯一入口

    async def ping(self) -> bool:
        try:
            await asyncio.to_thread(utility.get_server_version, using=self._alias)
            return True
        except Exception:
            return False
```

### MQ 客户端（`app/clients/mq.py`）

```python
@dataclass
class MQMessage:
    id: str
    data: dict
    stream: str

class AbstractMQClient(AbstractClient):
    @abstractmethod
    async def publish(self, stream: str, data: dict) -> str: ...
    @abstractmethod
    async def consume(self, stream: str, group: str, consumer: str, count: int = 10) -> AsyncIterator[MQMessage]: ...
    @abstractmethod
    async def ack(self, stream: str, group: str, message_id: str) -> None: ...
    @abstractmethod
    async def ensure_group(self, stream: str, group: str) -> None:
        """创建 Consumer Group（幂等）"""

class RedisStreamsMQClient(AbstractMQClient):
    """Redis Streams 初版，M4 可换 RabbitMQ/Kafka"""
```

### AbstractLLMClient（`app/clients/llm.py`）

```python
class AbstractLLMClient(AbstractClient):
    """M4 接入具体实现（OpenAI/Ollama/vLLM），此处预留熔断器包裹点"""
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
```

---

## Phase 4 — 统一异常处理

错误响应格式（含 `trace_id` 与日志保持一致）：

```json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Milvus connection failed",
    "request_id": "uuid4",
    "trace_id": null
  }
}
```

---

## Phase 5 — 健康检查

**端点：** `GET /health/live`（存活）, `GET /health/ready`（就绪，并发检查所有依赖）

```python
async def check_milvus(client: MilvusClient) -> tuple[str, str]:
    ok = await client.ping()   # 内部调用 utility.get_server_version()
    return "milvus", "ok" if ok else "error: version check failed"

async def check_mq(client: AbstractMQClient) -> tuple[str, str]:
    ok = await client.ping()   # xlen stream
    return "mq", "ok" if ok else "error: stream unreachable"
```

响应示例（`degraded` = 部分依赖不可用，服务仍运行）：

```json
{
  "status": "degraded",
  "kb_version": "v1",
  "checks": {
    "postgres": "ok",
    "redis": "ok",
    "milvus": "ok",
    "mq": "error: stream unreachable"
  }
}
```

---

## Phase 6 — 主应用组装

**文件：** `app/__init__.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup（顺序连接，失败时逆序断开已连接的）
    for name, client in named_clients:
        await client.connect()
        setattr(app.state, name, client)

    # LLM 三类信号量
    app.state.llm_semaphores = {
        "interactive": asyncio.Semaphore(settings.llm_semaphore_limits.interactive),
        "background":  asyncio.Semaphore(settings.llm_semaphore_limits.background),
        "audit":       asyncio.Semaphore(settings.llm_semaphore_limits.audit),
    }
    yield
    # Shutdown（逆序）
```

LLM 信号量依赖注入：

```python
LLMCategory = Literal["interactive", "background", "audit"]

def get_llm_semaphore(category: LLMCategory = "interactive"):
    def _inner(request: Request) -> asyncio.Semaphore:
        return request.app.state.llm_semaphores[category]
    return Depends(_inner)

# 路由用法
@router.post("/query")
async def query(sem: asyncio.Semaphore = get_llm_semaphore("interactive")):
    async with sem:
        ...
```

---

## Phase 7 — 测试策略

所有客户端用 `AsyncMock`，不发真实网络请求。集成测试（需 docker-compose）通过 `pytest -m integration` 单独运行。

```python
@pytest.fixture
def healthy_clients():
    postgres = AsyncMock(); postgres.ping.return_value = True
    redis    = AsyncMock(); redis.ping.return_value    = True
    milvus   = AsyncMock(); milvus.ping.return_value   = True
    mq       = AsyncMock(); mq.ping.return_value       = True
    return {"postgres": postgres, "redis": redis, "milvus": milvus, "mq": mq}

@pytest.fixture
def failing_clients():
    postgres = AsyncMock(); postgres.ping.return_value  = False
    redis    = AsyncMock(); redis.ping.side_effect      = ConnectionError("refused")
    milvus   = AsyncMock(); milvus.ping.return_value    = False
    mq       = AsyncMock(); mq.ping.return_value        = False
    return {"postgres": postgres, "redis": redis, "milvus": milvus, "mq": mq}
```

---

## Graph RAG 预留

### ChunkSchema（`app/models/chunk.py`）

```python
class ChunkSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    section_id: str
    chunk_index: int
    parent_chunk_id: Optional[str] = None   # 层级检索预留（Graph RAG）
    content_hash: str                        # 去重 & 版本追踪
    version: str                             # 对应 knowledge_base_version
    content: str
    embedding: Optional[list[float]] = None
```

### RetrievalStrategy Protocol（`app/models/retrieval.py`）

```python
@runtime_checkable
class RetrievalStrategy(Protocol):
    async def retrieve(self, query: str, top_k: int, **kwargs) -> list[ChunkSchema]: ...

class HybridRetriever:
    def __init__(self, strategies: list[RetrievalStrategy]) -> None: ...
    # RRF 融合排序在 M2 中实现
```

### LangGraph 节点占位（`app/routers/agent.py`）

```python
# LangGraph 节点注册顺序（M4 实现）:
# 1. query_rewrite      — 查询重写
# 2. retrieval          — 混合检索（调用 HybridRetriever）
# 3. entity_extraction  — pass-through（Graph RAG 预留）
# 4. rerank             — LLM 重排序
# 5. generate           — 流式输出
# 6. cache_write        — 写入 RAG 缓存
```

---

## 完整模块依赖图

```
M0  ←── 本次实现（基础架构）
 ├── M1  知识库（文档解析、向量化、HNSW）
 │    └── M6  一致性（KB 更新 → Redis 缓存失效）
 ├── M2  混合检索（BM25 + HNSW + RRF + 重排序）
 │    └── M3  RAG 缓存（Redis 跨用户）
 │         └── M4  Agent 编排（LangGraph + MQ + 信号量 + 流式输出）
 │              └── M7  Skill 封装
 └── M5  分层记忆（近期对话 + 长期摘要 + 静态知识向量化）
```

---

## 风险评估

| 风险 | 可能性 | 缓解方案 |
|------|--------|---------|
| pymilvus 同步 SDK 阻塞事件循环 | 高 | `asyncio.to_thread` 包装所有 pymilvus 调用 |
| Redis Streams Consumer Group 首次创建时 stream 不存在 | 中 | `ensure_group` 用 `MKSTREAM` flag 自动建流 |
| `llm_semaphores` 被错误 category 字符串访问 | 低 | `Literal` 类型 + 启动时验证 keys |
| Milvus alias 未注册时 `get_server_version()` 报错不清晰 | 中 | `connect` 时验证 alias 已注册，失败给明确错误 |
| `increment_with_ttl` Lua script 在 Redis Cluster 中跨 slot | 低 | 单节点无问题；Cluster 模式需 hash tag |

---

## 复杂度估时

| 阶段 | 估时 |
|------|------|
| Phase 1 配置 | 1.5h |
| Phase 2 日志 + 中间件注册 | 2h |
| Phase 3 五个客户端抽象 | 4.5h |
| Phase 4 异常处理 | 1h |
| Phase 5 健康检查 | 1h |
| Phase 6 主应用组装 | 1.5h |
| Phase 7 测试（AsyncMock 策略） | 3h |
| **合计** | **~14.5h** |
