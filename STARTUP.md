# 启动指南

## 环境要求
- Python 3.11+
- Docker Desktop（启动后等任务栏图标变绿）

---

## 首次启动

### 1. 安装依赖
```bash
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
# macOS/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

编辑 `.env`，填入必填项：
```
OPENAI_API_KEY=sk-...
# 使用第三方代理时额外填写：
# OPENAI_BASE_URL=https://api.ofox.ai/v1
```

---

## 启动方式

### 方式一：一键脚本（Windows 推荐）

```bat
start.bat   # 启动基础设施 + 后台运行应用
stop.bat    # 停止全部
```

- 自动检测 `.env` 是否存在，没有则从 `.env.example` 创建并提示填写 `OPENAI_API_KEY`
- uvicorn 在后台运行，日志写入 `logs\uvicorn.log`
- PID 保存在 `.uvicorn.pid`，`stop.bat` 用于精确停止

### 方式二：Make（macOS/Linux，或 Windows 安装 GNU Make 后）

```bash
make dev              # 启动基础设施 + 前台运行应用（支持热重载）
make down             # 停止基础设施
```

其他常用命令：
```bash
make install          # 安装依赖
make up               # 仅启动基础设施
make test             # 运行单元测试
make test-integration # 运行所有测试（含集成测试）
make lint             # ruff check + mypy
make format           # ruff format + fix
make logs             # 查看容器日志
make clean            # 清理缓存文件
```

### 方式三：手动

```bash
# 1. 启动基础设施（等待所有服务 healthy，约 30-60 秒）
docker compose up -d --wait

# 2. 启动应用
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 验证

```bash
# 基础健康
curl http://localhost:8000/health

# 全量就绪检查（含 LLM 连通性）
curl http://localhost:8000/health/ready
```

| 地址 | 说明 |
|------|------|
| http://localhost:8000 | 前端对话页面 |
| http://localhost:8000/docs | OpenAPI 文档 |
| http://localhost:8000/health | 健康检查 |

---

## 核心 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/knowledge-base/documents` | 上传文档（form-data `file`）|
| GET  | `/knowledge-base/documents/{doc_id}/status` | 查询文档状态 |
| POST | `/agent/query` | 提交问答任务 |
| GET  | `/agent/jobs/{job_id}` | 查询任务结果 |
| GET  | `/agent/stream/{job_id}` | SSE 流式获取结果 |
| GET  | `/retrieval/search` | 直接检索 |
| GET  | `/cache/stats` | 缓存统计 |

**上传文档：**
```bash
curl -X POST http://localhost:8000/knowledge-base/documents \
  -F "file=@your_document.pdf"
```

**问答（轮询）：**
```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "query": "文档讲了什么？", "top_k": 5}'

# 轮询结果
curl http://localhost:8000/agent/jobs/{job_id}
```

**问答（SSE 流式）：**
```bash
curl -N http://localhost:8000/agent/stream/{job_id}
```

---

## 停止

**Windows：**
```bat
stop.bat
```

**手动：**
```bash
# 仅停应用：Ctrl+C

# 停基础设施（保留数据）
docker compose stop

# 停基础设施并清除数据
docker compose down -v
```
