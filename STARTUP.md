# 启动指南

## 环境要求
- Python 3.11+
- Docker Desktop（启动后等任务栏图标变绿）

## 首次启动

### 1. 安装依赖
```bash
pip install -e ".[dev]"
```

### 2. 配置环境变量
```bash
cp .env.example .env
```
编辑 `.env`，填入必填项：
```
OPENAI_API_KEY=sk-...
# 使用第三方代理时额外填写：
# OPENAI_BASE_URL=https://api.ofox.ai/v1
```

### 3. 启动基础设施
```bash
docker compose up -d
```
等待所有容器变为 healthy（约 30 秒）：
```bash
docker compose ps   # 确认 5 个服务全部 healthy
```

### 4. 启动应用
```bash
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/health` 确认返回 `{"status":"ok"}`。

---

## 日常启动（非首次）

```bash
docker compose up -d   # 启动基础设施
python main.py         # 启动应用
```

---

## 验证

```bash
# 基础健康
curl http://localhost:8000/health

# 全量就绪检查（含 LLM 连通性）
curl http://localhost:8000/health/ready

# OpenAPI 文档
open http://localhost:8000/docs
```

---

## 核心 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/knowledge-base/documents` | 上传文档（form-data `file`）|
| GET  | `/knowledge-base/documents/{doc_id}/status` | 查询文档状态 |
| POST | `/agent/query` | 提交问答任务 |
| GET  | `/agent/jobs/{job_id}` | 查询任务结果 |
| GET  | `/retrieval/search` | 直接检索 |
| GET  | `/cache/stats` | 缓存统计 |

**上传文档：**
```bash
curl -X POST http://localhost:8000/knowledge-base/documents \
  -F "file=@your_document.pdf"
```

**问答：**
```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "query": "文档讲了什么？", "top_k": 5}'

# 轮询结果
curl http://localhost:8000/agent/jobs/{job_id}
```

---

## 停止

```bash
# 仅停应用：Ctrl+C

# 停基础设施（保留数据）
docker compose stop

# 停基础设施并清除数据
docker compose down -v
```
