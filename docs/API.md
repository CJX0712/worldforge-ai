# WorldForge AI HTTP API 参考

- 对应架构: `docs/ARCHITECTURE.md` 第 8 节 (HTTP API 设计)
- 相关决策: `docs/decisions/ADR-0001.md` (零依赖核心), `docs/decisions/ADR-0002.md` (确定性)
- 适用范围: `src/worldforge/serving/`

---

## 0. 结论先行

| 项 | 说明 |
| --- | --- |
| 网关实现 | **stdlib `http.server.ThreadingHTTPServer`**, 零第三方依赖, 裸 CPython 3.11+ 直接启动 |
| 可选加速 | 装了 `uvicorn` + FastAPI 时可用 `WORLDFORGE_SERVER=uvicorn` 切换, 未装则回退 stdlib |
| 默认监听 | `127.0.0.1:8000`, 由 `WORLDFORGE_HOST` / `WORLDFORGE_PORT` 覆盖 |
| 环境变量前缀 | 规范形式 `WORLDFORGE_*`; `WF_*` 为等价别名 (见 ADR-0001 D6) |
| 认证 | 默认关闭; 设置 `WORLDFORGE_API_KEY` 后校验 `Authorization: Bearer <key>` |
| 编码 | 请求与响应一律 `application/json; charset=utf-8`; SSE 为 `text/event-stream` |

启动方式:

| 方式 | 命令 |
| --- | --- |
| 模块启动 | `python -m worldforge.serving` |
| 脚本启动 | `python scripts/smoke.py` (自动起服务 + 打全流程 + 关服务) |
| 指定端口 | `WORLDFORGE_PORT=8080 python -m worldforge.serving` |
| 严格模式 | `WORLDFORGE_STRICT=on python -m worldforge.serving` (后端缺失直接启动失败, 不静默回退) |

---

## 1. 通用约定

| 项 | 约定 |
| --- | --- |
| 路径前缀 | `/v1` (`/health` 除外) |
| 请求体 | JSON object; 未知字段一律忽略, 不做严格 reject |
| 响应体 | JSON object; 字段顺序不作为契约, 客户端按 key 取值 |
| 时间戳字段 | 仅 `created` (Unix 秒, int) 出现在响应中; 延迟类字段只在 `/v1/metrics` 与 `/v1/traces/{id}?wall=1` |
| trace | 每个响应带 `trace_id`, 可用于 `GET /v1/traces/{trace_id}` 回放 |
| 幂等 | `POST /v1/ingest` 幂等: 同 `doc_id` 覆盖写入, 内部先 `drop_document` |
| 限流 | 默认关闭; 开启后超限返回 429 |

### 1.1 错误信封

所有非 2xx 响应统一结构:

```json
{
  "error": {
    "code": "doc_not_found",
    "message": "document 'd9' does not exist",
    "trace_id": "tr_7f3a1c9e"
  }
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `error.code` | string | snake_case 稳定错误码, 客户端可按此分支 |
| `error.message` | string | 人类可读; **不作为判断依据**, 文案可改 |
| `error.trace_id` | string | 用于 `GET /v1/traces/{trace_id}` 排障 |

### 1.2 错误码总表

| HTTP | `code` | 触发条件 | 出现端点 |
| --- | --- | --- | --- |
| 400 | `invalid_request` | 必填字段缺失 / 参数取值非法 | 全部写端点 |
| 401 | `unauthorized` | `WORLDFORGE_API_KEY` 已设但 Bearer 缺失或不匹配 | 全部 |
| 403 | `reset_not_allowed` | `WORLDFORGE_ALLOW_RESET=off` 时调用 reset | `/v1/admin/reset` |
| 404 | `doc_not_found` | `doc_id` 不存在 | `DELETE /v1/documents/{id}` |
| 404 | `trace_not_found` | `trace_id` 不存在或已被环形缓冲淘汰 | `/v1/traces/{id}` |
| 404 | `unknown_path` | 路径未注册 | 全部 |
| 413 | `payload_too_large` | 文档数超过 `WORLDFORGE_MAX_DOCS` 或单文档超长 | `/v1/ingest` |
| 422 | `schema_invalid` | JSON 解析失败或字段类型不符 | 全部写端点 |
| 429 | `rate_limited` | 开启限流后超限 | 全部 |
| 500 | `internal_error` | 未捕获异常 (应被视为 bug) | 全部 |
| 503 | `backend_unavailable` | `WORLDFORGE_STRICT=on` 且所需后端缺失 | 全部 |

注意: 评估未过门限**不返回错误码**, `/v1/evaluate` 仍返回 200 且 `passed=false`。进程内 CLI 才抛 `EvalGateError`。

---

## 2. 端点清单

| # | 方法 | 路径 | 用途 |
| --- | --- | --- | --- |
| 1 | GET | `/health` | 存活与后端降级状态 |
| 2 | POST | `/v1/ingest` | 摄入文档 (先 drop 再写) |
| 3 | DELETE | `/v1/documents/{doc_id}` | 删除文档及其全部分块 |
| 4 | GET | `/v1/documents` | 列出 doc_id |
| 5 | POST | `/v1/query` | 检索 (不生成) |
| 6 | POST | `/v1/chat/completions` | RAG / Agent 问答, OpenAI 兼容, 支持 SSE |
| 7 | POST | `/v1/evaluate` | 跑评估套件并返回 scorecard |
| 8 | GET | `/v1/metrics` | 指标导出 (JSON / Prometheus) |
| 9 | GET | `/v1/traces/{trace_id}` | 单条 trace 的 span 回放 |
| 10 | POST | `/v1/admin/reset` | 清空索引 (需显式确认) |

---

## 3. GET /health

零依赖探活, 不触发任何后端 I/O。

### 响应 200

```json
{
  "status": "degraded",
  "version": "0.1.0",
  "backends": {
    "embedder": "hashing",
    "vectorstore": "memory",
    "lexical": "bm25",
    "reranker": "lexical",
    "llm": "mock",
    "docstore": "memory",
    "planner": "rule",
    "chunker": "recursive",
    "observer": "memory"
  },
  "degraded": ["vectorstore"],
  "index": {"docs": 12, "chunks": 87}
}
```

| 字段 | 说明 |
| --- | --- |
| `status` | `ok` 无降级; `degraded` 至少一个后端回退到默认实现 |
| `backends` | 实际生效的实现名 (不是请求名) -- 用于发现「以为在用 FAISS 实际在跑暴力检索」 |
| `degraded` | 发生回退的能力 tag 列表 |
| `index` | 当前索引规模 |

### 错误

| 码 | code |
| --- | --- |
| 500 | `internal_error` |

### curl

```bash
curl -s http://127.0.0.1:8000/health | python -c "import sys,json;d=json.load(sys.stdin);print(d['status'], d['degraded'])"
```

---

## 4. POST /v1/ingest

摄入文档。语义为 **upsert**: 同 `doc_id` 先 `drop_document` 再写入, 不会产生孤儿分块。

### 请求

```json
{
  "documents": [
    {
      "doc_id": "d1",
      "text": "WorldForge 是一个可插拔的端到端 AI 平台。它把检索、推理、编排拆成可替换的层。",
      "meta": {"lang": "zh", "source": "readme"}
    }
  ],
  "chunk": true
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `documents` | array | 是 | 长度 1..`WORLDFORGE_MAX_DOCS` (默认 64) |
| `documents[].doc_id` | string | 是 | 幂等键 |
| `documents[].text` | string | 是 | 空或纯空白拒绝 (400) |
| `documents[].meta` | object | 否 | 等值过滤用, 值只支持 string/number/bool |
| `chunk` | bool | 否 | 默认 `true`; `false` 时整篇作为单块 |

### 响应 200

```json
{
  "docs": 1,
  "chunks": 7,
  "skipped": 0,
  "trace_id": "tr_7f3a1c9e"
}
```

| 字段 | 说明 |
| --- | --- |
| `docs` | 成功写入文档数 |
| `chunks` | 产生的分块数 |
| `skipped` | 因超 `WORLDFORGE_MAX_CHUNKS` (默认 5000) 被截断的分块数 |

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 400 | `invalid_request` | `documents` 缺失 / 空数组 / `text` 为空 |
| 413 | `payload_too_large` | 文档数或总字符数超限 |
| 422 | `schema_invalid` | `documents` 非数组 / `doc_id` 非字符串 |
| 500 | `internal_error` | 写入过程异常 |

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"doc_id":"d1","text":"WorldForge 是一个可插拔的端到端 AI 平台。","meta":{"lang":"zh"}}]}'
```

---

## 5. DELETE /v1/documents/{doc_id}

同步删除 DocStore + VectorStore + LexicalIndex 三处, 三者任一失败即整体回滚报错。

### 响应 200

```json
{"doc_id": "d1", "deleted_chunks": 7}
```

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 404 | `doc_not_found` | `doc_id` 不存在 |
| 500 | `internal_error` | 删除中断 |

### curl

```bash
curl -s -X DELETE http://127.0.0.1:8000/v1/documents/d1
```

---

## 6. GET /v1/documents

列出 `doc_id`, 按字典序稳定排序。

### Query

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `limit` | 100 | 1..1000 |
| `offset` | 0 | >= 0 |

### 响应 200

```json
{"doc_ids": ["d1", "d2"], "total": 2}
```

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 400 | `invalid_request` | `limit` / `offset` 越界或非整数 |
| 500 | `internal_error` | 存储异常 |

### curl

```bash
curl -s "http://127.0.0.1:8000/v1/documents?limit=10&offset=0"
```

---

## 7. POST /v1/query

只检索不生成。用于调试召回质量与做检索类评估。

### 请求

```json
{
  "query": "WorldForge 的检索层怎么分层?",
  "top_k": 20,
  "mode": "hybrid",
  "where": {"lang": "zh"}
}
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `query` | string | - | 必填; 空串拒绝 |
| `top_k` | int | `WORLDFORGE_TOP_K` (20) | 单路召回条数 |
| `mode` | string | `hybrid` | `hybrid` / `dense` / `sparse` |
| `where` | object | null | 等值过滤, AND 语义 |

### 响应 200

```json
{
  "hits": [
    {
      "chunk_id": "d1#0003",
      "doc_id": "d1",
      "score": 0.8312,
      "source": "rerank",
      "text": "它把检索、推理、编排拆成可替换的层。",
      "meta": {"lang": "zh", "source": "readme"}
    }
  ],
  "trace_id": "tr_7f3a1c9e"
}
```

| 字段 | 说明 |
| --- | --- |
| `hits` | 按 `(-score, chunk_id)` 稳定排序 |
| `hits[].source` | `dense` / `sparse` / `fused` / `rerank`; 融合后统一为 `rerank` |
| `hits[].text` | 已单行化 (无内嵌换行) |

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 400 | `invalid_request` | `query` 缺失 / 空 / `top_k` <= 0 / `mode` 非法 |
| 422 | `schema_invalid` | 类型不符 |
| 500 | `internal_error` | 检索异常 |
| 503 | `backend_unavailable` | `WORLDFORGE_STRICT=on` 且后端缺失 |

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"检索层怎么分层?","top_k":10,"mode":"hybrid"}'
```

---

## 8. POST /v1/chat/completions

OpenAI 兼容端点。默认走 RAG; `WORLDFORGE_PLANNER` 非 `rule` 时走 Agent 分支 (规划 -> 工具调用 -> 汇总)。

### 请求

```json
{
  "model": "worldforge-1",
  "messages": [
    {"role": "system", "content": "只依据证据回答。"},
    {"role": "user", "content": "WorldForge 的检索层怎么分层?"}
  ],
  "temperature": 0,
  "max_tokens": 512,
  "stream": false,
  "worldforge": {
    "mode": "rag",
    "top_k": 5
  }
}
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | string | - | 任意字符串, 服务端忽略取值, 回显到响应 |
| `messages` | array | - | `role` in `system`/`user`/`assistant`/`tool`; 至少一条 |
| `temperature` | float | 0 | 评估与默认路径**必须**为 0 |
| `max_tokens` | int | 512 | |
| `stream` | bool | false | true 时返回 SSE |
| `worldforge.mode` | string | `rag` | `rag` / `auto` / `agent` |
| `worldforge.top_k` | int | `WORLDFORGE_RERANK_TOP_K` (5) | 送入 LLM 的证据条数 |

### 响应 200 (非流式)

```json
{
  "id": "chatcmpl-01HQ8Z3F2K",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "worldforge-1",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "检索层分为稀疏与稠密两路 [d1#0003]。"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 213,
    "completion_tokens": 24,
    "total_tokens": 237
  },
  "worldforge": {
    "evidence": [
      {"chunk_id": "d1#0003", "doc_id": "d1", "score": 0.8312, "text": "它把检索、推理、编排拆成可替换的层。"}
    ],
    "verification": {
      "grounding": 0.87,
      "faithfulness": 0.79,
      "unsupported": []
    },
    "tool_calls": [],
    "degraded": ["vectorstore"],
    "trace_id": "tr_7f3a1c9e"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `choices[0].finish_reason` | `stop` / `length` / `tool_calls` |
| `worldforge.evidence` | 实际送入 LLM 的证据, 已单行化 |
| `worldforge.verification` | 接地核验结果; `WORLDFORGE_VERIFY=off` 时该字段为 `null` |
| `worldforge.tool_calls` | Agent 分支下非空: `{"name","args","ok","output"}` |
| `worldforge.degraded` | 本次请求实际发生回退的能力 |

### 响应 200 (流式, SSE)

`Content-Type: text/event-stream`。

```
data: {"id":"chatcmpl-01HQ8Z3F2K","object":"chat.completion.chunk","created":1760000000,"model":"worldforge-1","choices":[{"index":0,"delta":{"role":"assistant","content":"检索层"}}]}

data: {"id":"chatcmpl-01HQ8Z3F2K","object":"chat.completion.chunk","created":1760000000,"model":"worldforge-1","choices":[{"index":0,"delta":{"content":"分为稀疏与稠密两路"}}]}

data: {"id":"chatcmpl-01HQ8Z3F2K","object":"chat.completion.chunk","created":1760000000,"model":"worldforge-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"worldforge":{"verification":{"grounding":0.87},"trace_id":"tr_7f3a1c9e"}}

data: [DONE]
```

| 规则 | 说明 |
| --- | --- |
| 帧分隔 | 每帧以空行结束; 最后一帧固定 `data: [DONE]` |
| 末帧 | 附带 `worldforge.verification` 与 `trace_id`, 中间帧不重复 |
| 确定性 | 流式输出**不进**确定性断言; 断言只对非流式路径生效 |

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 400 | `invalid_request` | `messages` 为空 / `role` 非法 / `temperature` < 0 |
| 422 | `schema_invalid` | `messages` 非数组 |
| 429 | `rate_limited` | 限流开启且超限 |
| 500 | `internal_error` | 生成或核验异常 |
| 503 | `backend_unavailable` | `WORLDFORGE_STRICT=on` 且 LLM 后端缺失 |

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"worldforge-1","messages":[{"role":"user","content":"检索层怎么分层?"}],"temperature":0}'
```

流式:

```bash
curl -N -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"worldforge-1","messages":[{"role":"user","content":"检索层怎么分层?"}],"stream":true}'
```

Python 客户端 (注意 `trust_env=False`, 见 `ARCHITECTURE.md` FM-11):

```python
import httpx
with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=30.0) as c:
    r = c.post("/v1/chat/completions", json={
        "model": "worldforge-1",
        "messages": [{"role": "user", "content": "检索层怎么分层?"}],
        "temperature": 0,
    })
    print(r.json()["choices"][0]["message"]["content"])
```

---

## 9. POST /v1/evaluate

跑评估套件并返回 scorecard。**门限未过仍返回 200**, 客户端读 `passed` 字段判定。

### 请求

```json
{
  "suite": "default",
  "cases": null,
  "record_baseline": false
}
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `suite` | string | `default` | `default` / `retrieval` / `rag` / `agent` |
| `cases` | array \| null | null | null 表示跑套件全量; 传数组则只跑指定 `case_id` |
| `record_baseline` | bool | false | true 时把本次结果写入 `baseline.json` (需 `WORLDFORGE_ALLOW_BASELINE_WRITE=on`) |

### 响应 200

```json
{
  "suite": "default",
  "passed": false,
  "metrics": {
    "doc_hit_rate@5": 0.83,
    "recall@10": 0.71,
    "mrr": 0.62,
    "grounding": 0.87,
    "faithfulness": 0.79,
    "tool_route_accuracy": 0.9,
    "tool_call_success": 1.0
  },
  "agent_metrics": {
    "tool_route_accuracy": 0.9,
    "tool_call_success": 1.0,
    "cases": 20
  },
  "gates": [
    {"metric": "doc_hit_rate@5", "value": 0.83, "threshold": 0.7885, "floor": 0.60, "hard": true, "ok": true},
    {"metric": "grounding", "value": 0.87, "threshold": 0.8265, "floor": 0.55, "hard": true, "ok": true},
    {"metric": "mrr", "value": 0.62, "threshold": 0.60, "floor": 0.50, "hard": true, "ok": false}
  ],
  "warnings": ["ndcg@10 dropped 3.1% vs baseline"],
  "perf": {"latency_p50_ms": 41.2, "latency_p95_ms": 118.7},
  "trace_id": "tr_7f3a1c9e"
}
```

| 字段 | 说明 |
| --- | --- |
| `passed` | 所有 `hard` 门限的 `ok` 全为 true |
| `metrics` | 检索与 RAG 聚合指标 (不含工具路由, 见第 9.1 节) |
| `agent_metrics` | 工具路由与执行指标, **单独计数上报** |
| `gates[].threshold` | `max(floor, baseline * 0.95)` |
| `perf` | 延迟分位, 不参与 `gates` |

### 9.1 两条强制约束

| 约束 | 说明 |
| --- | --- |
| 每个 case 全新管道 | 套件内每个用例都调用 `build_pipeline(env)` 新建, 禁止复用运行期单例管道, 否则索引与缓存残留会污染指标 |
| 工具路由类用例不进检索聚合 | `tool_route_accuracy` / `tool_call_success` 只在 `agent_metrics` 中单独计数, 不得混入 `metrics` 的平均值, 否则量纲不同会稀释检索指标 |

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 400 | `invalid_request` | 未知 `suite` / `cases` 引用不存在的 `case_id` |
| 403 | `reset_not_allowed` | `record_baseline=true` 但未开写入开关 |
| 422 | `schema_invalid` | 类型不符 |
| 500 | `internal_error` | 评估过程异常 |

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"suite":"retrieval"}'
```

---

## 10. GET /v1/metrics

### Query

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `format` | `json` | `json` / `prom` |

### 响应 200 (json)

```json
{
  "counters": {
    "http_requests_total": 128,
    "http_errors_total": 3,
    "embed_calls_total": 256,
    "tool_calls_total": 41
  },
  "gauges": {
    "index_docs": 12,
    "index_chunks": 87,
    "degraded_backends": 1
  },
  "histograms": {
    "http_latency_ms": {"p50": 41.2, "p95": 118.7, "p99": 203.4, "n": 128},
    "llm_latency_ms": {"p50": 12.4, "p95": 31.9, "p99": 48.1, "n": 64}
  }
}
```

### 响应 200 (prom)

```
# HELP worldforge_http_requests_total Total HTTP requests
# TYPE worldforge_http_requests_total counter
worldforge_http_requests_total 128
# HELP worldforge_http_latency_ms_p95 HTTP latency p95 in ms
# TYPE worldforge_http_latency_ms_p95 gauge
worldforge_http_latency_ms_p95 118.7
```

### 错误

| 码 | code |
| --- | --- |
| 400 | `invalid_request` (`format` 非法) |
| 500 | `internal_error` |

### curl

```bash
curl -s "http://127.0.0.1:8000/v1/metrics?format=prom"
```

---

## 11. GET /v1/traces/{trace_id}

回放单条 trace 的 span 序列。默认不返回墙钟字段。

### Query

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `wall` | `0` | `1` 时附带 `wall_ms` (仅排障用, 禁止进断言) |

### 响应 200

```json
{
  "trace_id": "tr_7f3a1c9e",
  "spans": [
    {"order": 0, "name": "http.query", "attrs": {"path": "/v1/query"}},
    {"order": 1, "name": "retrieve.dense", "attrs": {"top_k": 20}},
    {"order": 2, "name": "retrieve.sparse", "attrs": {"top_k": 20}},
    {"order": 3, "name": "fusion.rrf", "attrs": {"k": 60}},
    {"order": 4, "name": "rerank", "attrs": {"top_k": 5}}
  ],
  "events": [
    {"order": 0, "event": "intent.detected", "fields": {"intent": "qa"}},
    {"order": 1, "event": "evidence.lines", "fields": {"count": 5}}
  ]
}
```

`?wall=1` 时每个 span 额外带 `"wall_ms": 3.21`。

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 404 | `trace_not_found` | 不存在, 或已被 `WORLDFORGE_TRACE_BUFFER` (默认 1000) 环形缓冲淘汰 |
| 500 | `internal_error` | 缓冲损坏 |

### curl

```bash
curl -s "http://127.0.0.1:8000/v1/traces/tr_7f3a1c9e?wall=1"
```

---

## 12. POST /v1/admin/reset

清空全部索引与文档。默认需显式确认且需开关放行。

### 请求

```json
{"confirm": true}
```

### 响应 200

```json
{"cleared": {"docs": 12, "chunks": 87}}
```

### 错误

| 码 | code | 触发 |
| --- | --- | --- |
| 400 | `invalid_request` | `confirm` 不为 `true` |
| 403 | `reset_not_allowed` | `WORLDFORGE_ALLOW_RESET=off` (默认 off) |
| 500 | `internal_error` | 清理中断 |

### curl

```bash
WORLDFORGE_ALLOW_RESET=on
curl -s -X POST http://127.0.0.1:8000/v1/admin/reset \
  -H "Content-Type: application/json" \
  -d '{"confirm":true}'
```

---

## 13. 端到端冒烟序列

| 步 | 命令 | 期望 |
| --- | --- | --- |
| 1 | `curl -s /health` | `status` 为 `ok` 或 `degraded` |
| 2 | `POST /v1/ingest` 写入 2 篇文档 | `chunks` > 0 |
| 3 | `POST /v1/query` 用文档内的问题 | `hits[0].chunk_id` 属于预期文档 |
| 4 | `POST /v1/chat/completions` | `content` 非空且 `grounding` >= floor |
| 5 | `GET /v1/traces/{trace_id}` | `spans` 含 `retrieve.dense` 与 `fusion.rrf` |
| 6 | `POST /v1/evaluate` | `passed` 为 true (基线已回填时) |
| 7 | `GET /v1/metrics?format=prom` | 文本非空 |

等价于 `python scripts/smoke.py`, 该脚本内置 `httpx.Client(trust_env=False)`, 避免 Windows 系统代理导致 `WinError 10054`。

---

## 14. 版本与兼容

| 项 | 约定 |
| --- | --- |
| 版本前缀 | `/v1`; 破坏性变更走 `/v2`, 不原地改 |
| 字段增删 | 允许新增字段; 已有字段不得改名或改语义 |
| OpenAI 兼容 | `/v1/chat/completions` 的 OpenAI 标准字段与官方一致; 扩展字段一律放 `worldforge` 键下 |
| 错误码 | `error.code` 取值集合只增不改 |
