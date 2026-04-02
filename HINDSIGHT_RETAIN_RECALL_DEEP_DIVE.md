# Hindsight `api_retain` & `api_recall` 完整执行链深度解析

> **目标读者**：具备扎实后端/系统设计能力，尚未深入 AI Agent 框架源码的工程师。
> **方法论**：BFS 宏观流 → DFS 关键函数 → 真实运行示例 → 补充知识（论文/设计思想）

---

## 目录

1. [系统架构全景](#1-系统架构全景)
2. [BFS：api_retain 执行流](#2-bfs：api_retain-执行流)
3. [DFS：retain 关键机制深挖](#3-dfs：retain-关键机制深挖)
   - [3.1 Fact Extraction — LLM 结构化输出](#31-fact-extraction--llm-结构化输出)
   - [3.2 Delta Retain — 增量更新优化](#32-delta-retain--增量更新优化)
   - [3.3 Entity Resolver — 实体消歧](#33-entity-resolver--实体消歧)
   - [3.4 Link Creation — 四类记忆边](#34-link-creation--四类记忆边)
4. [BFS：api_recall 执行流](#4-bfs：api_recall-执行流)
5. [DFS：recall 关键机制深挖](#5-dfs：recall-关键机制深挖)
   - [5.1 四路并行检索](#51-四路并行检索)
   - [5.2 Reciprocal Rank Fusion (RRF)](#52-reciprocal-rank-fusion-rrf)
   - [5.3 Cross-Encoder 神经重排](#53-cross-encoder-神经重排)
   - [5.4 Combined Scoring — 时间感知打分](#54-combined-scoring--时间感知打分)
6. [真实运行示例](#6-真实运行示例)
7. [补充知识：论文/设计思想映射](#7-补充知识：论文设计思想映射)
8. [性能与容错设计](#8-性能与容错设计)

---

## 1. 系统架构全景

```
HTTP 请求
    │
    ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Router  (hindsight_api/api/http.py)     │
│  api_retain / api_recall                         │
└────────────┬────────────────────────────────────┘
             │  app.state.memory  (单例注入)
             ▼
┌─────────────────────────────────────────────────┐
│  MemoryEngine  (engine/memory_engine.py)         │
│  retain_batch_async() / recall_async()           │
└──────┬──────────────────────────┬───────────────┘
       │ RETAIN                   │ RECALL
       ▼                          ▼
┌─────────────────┐    ┌──────────────────────────┐
│ retain/          │    │ search/                   │
│ orchestrator.py  │    │ retrieval.py (4路并行)    │
│ ├ fact_extraction│    │ ├ semantic (pgvector HNSW)│
│ ├ embedding      │    │ ├ bm25 (全文检索)         │
│ ├ entity_resolver│    │ ├ graph (BFS spreading)   │
│ └ link_creation  │    │ └ temporal (时间扩散)     │
└──────────────────┘    │ fusion.py (RRF)           │
                        │ reranking.py (交叉编码器) │
                        └──────────────────────────┘
                                   │
                              PostgreSQL
                            (pgvector + BM25)
```

**数据模型三层：**

| 类型 | 说明 | 举例 |
|------|------|------|
| `world` | 客观世界事实 | "Alice 在 Google 工作" |
| `experience` | 第一人称经历/行动 | "我昨天修复了那个 API 慢查询" |
| `mental_model` | 合并后的高阶认知 | "用户偏好函数式编程风格" |

---

## 2. BFS：api_retain 执行流

### 入口：`POST /v1/default/banks/{bank_id}/memories/retain`

```
api/http.py : api_retain()
    │
    ├─ 1. 按 strategy 分组请求 items
    │      strategy_groups: dict[str|None, list[dict]]
    │
    ├─ 2. 路由判断
    │      async=true  → submit_async_retain()  → 任务队列  → 立即返回 operation_id
    │      async=false → retain_batch_async()   → 同步等待  → 返回 usage
    │
    └─ 3. 返回 RetainResponse
           {success, bank_id, items_count, async, operation_id?, usage?}
```

### 核心路径：同步 retain

```
memory_engine.py : retain_batch_async()
    │
    ├─ 认证租户  _authenticate_tenant(request_context)
    ├─ 可选扩展钩子  _operation_validator.validate_retain()
    ├─ 自动分批（超过 retain_batch_tokens 阈值时）
    │
    └─ retain/orchestrator.py : retain_batch()
            │
            ├─ Step 1: 获取 bank profile（agent_name）
            ├─ Step 2: Delta retain 检查（复用未改变的 chunk）
            │          ↓ 如不命中，走全量路径
            ├─ Step 3: _extract_and_embed()
            │    ├─ fact_extraction.extract_facts_from_contents()  ← LLM
            │    └─ embedding_processing.generate_embeddings_batch() ← 向量化
            │
            └─ Step 4: DB 事务（acquire_with_retry + deadlock retry）
                 ├─ fact_storage.handle_document_tracking()  # 文档 upsert
                 ├─ chunk_storage.store_chunks_batch()       # 存原始 chunk
                 ├─ fact_storage.insert_facts_batch()        # 存 memory_units
                 ├─ entity_processing.process_entities_batch()
                 ├─ link_creation.create_temporal_links_batch()
                 ├─ link_creation.create_semantic_links_batch()
                 ├─ entity_processing.insert_entity_links_batch()
                 └─ link_creation.create_causal_links_batch()
```

---

## 3. DFS：retain 关键机制深挖

### 3.1 Fact Extraction — LLM 结构化输出

**文件：** `engine/retain/fact_extraction.py`

#### 设计思路：5W 结构化提取

Hindsight 不是把原始文本直接向量化存储，而是先让 LLM 将内容解构成 **5W 维度的结构化事实**：

```python
# fact_extraction.py : ExtractedFact
class ExtractedFact(BaseModel):
    what: str   # 核心事实（1-2句话）
    when: str   # 时间（ISO 日期 or "N/A"）
    where: str  # 地点
    who: str    # 涉及人员（解析共指：如"我室友 Emily"→"Emily，用户的室友"）
    why: str    # 背景/动机/情感

    fact_type: Literal["world", "assistant"]  # world=客观, assistant=第一人称
    occurred_start: str | None  # ISO 时间戳
    occurred_end: str | None
    entities: list[Entity] | None
    causal_relations: list[FactCausalRelation] | None  # 与之前 fact 的因果关系
```

最终存储格式（`Fact.build_fact_text()`）：
```
"Alice 与 Sarah 在天台花园完婚，50 位宾客出席 | Involving: Alice(用户大学室友), Sarah | 用户深受感动，曾梦想在户外举行婚礼"
```

这比直接存原始文本的好处：
1. **密度更高**：噪音被过滤，语义更集中
2. **时间提取**：LLM 自动解析 "last night"→具体日期
3. **实体识别**：为后续图构建打基础
4. **因果链**：`target_index < 当前 index`，防止幻觉引用

#### Prompt 设计（不同 mode）

```python
# fact_extraction.py 中有多种提取模式：
# - concise (默认): 精简提取
# - verbose: 详细提取，捕捉所有细节
# - verbatim: 保留原文，只提取元数据
# - chunks: 直接分块，不调用 LLM
```

**Token 溢出处理**：
```python
# 当 LLM 输出超出限制时，抛出 OutputTooLongError
# extract_facts_from_contents() 自动将超长 content 拆分后重试
```

#### 并行提取

```python
# 多个 content 并发调用 LLM（asyncio.gather）
# 每个 chunk 一次独立请求，最大并发受全局 semaphore 控制
tasks = [extract_facts(c, llm_config, ...) for c in contents]
results = await asyncio.gather(*tasks)
```

---

### 3.2 Delta Retain — 增量更新优化

**文件：** `engine/retain/orchestrator.py : _try_delta_retain()`

当同一个 `document_id` 的文档被重新 retain 时（常见于 agent 反复更新知识），**Delta Retain** 比较 chunk 内容哈希：

```
旧版文档 chunks:  [chunk_A_hash, chunk_B_hash, chunk_C_hash]
新版文档 chunks:  [chunk_A_hash, chunk_B_NEW_hash, chunk_C_hash]
                           ↑ 只有这个变了
Delta Retain:
  - chunk_A: 复用现有 memory_units，跳过 LLM+Embedding
  - chunk_B: 删除旧 units → 重新提取 → 重新向量化
  - chunk_C: 复用现有 memory_units
```

**好处**：对大型文档（如长篇对话记录）反复更新时，节省 70-80% 的 LLM token 和 embedding 成本。

---

### 3.3 Entity Resolver — 实体消歧

**文件：** `engine/entity_resolver.py`

同一实体在不同 fact 中可能有不同写法（"Alice"、"我室友 Alice"、"Alice Chen"），Entity Resolver 做实体消歧和归一化：

```python
# 两种查找策略（配置驱动）：
# "full"    → 加载 bank 的全部实体到内存，做精确匹配
# "trigram" → 使用 pg_trgm 索引做相似度查找（适合大型 bank）

# 实体解析后，建立 entity_links：
# memory_unit_id → entity_id（多对多关系）
```

实体共现统计（co-occurrence）在后台异步更新，用于计算实体重要性。

---

### 3.4 Link Creation — 四类记忆边

**文件：** `engine/retain/link_utils.py`

| 边类型 | 连接方式 | 用途 |
|--------|---------|------|
| `temporal` | 时间上相邻的 memory unit | 构建时间线 |
| `semantic` | 向量相似度 > 阈值（HNSW） | 语义关联 |
| `entity` | 共享同一个实体 | 实体图谱 |
| `causal` | LLM 提取的 `caused_by` 关系 | 因果推理 |

这四类边共同构成了 Hindsight 的**异构记忆图（Heterogeneous Memory Graph）**，为 Graph Retrieval 提供导航结构。

```python
# link_creation.py
# 因果链权重（BFS 时的 activation 增强）：
# "causal"     → 2.0x boost
# "enables"    → 1.5x boost
# "prevents"   → 1.5x boost
```

---

## 4. BFS：api_recall 执行流

### 入口：`POST /v1/default/banks/{bank_id}/memories/recall`

```
api/http.py : api_recall()
    │
    ├─ 1. 验证 query token 长度（防止超长查询 DoS）
    │      max_query_tokens = config.recall_max_query_tokens
    │
    ├─ 2. 解析参数
    │      fact_types = request.types or ["world", "experience"]
    │      question_date = parse(request.query_timestamp)
    │      include_entities / include_chunks / include_source_facts
    │
    ├─ 3. 调用核心引擎
    │      core_result = await memory.recall_async(...)
    │
    ├─ 4. 转换响应
    │      MemoryFact → RecallResult（过滤内部 metrics）
    │      ChunkInfo  → ChunkData
    │
    └─ 5. 返回 RecallResponse
```

### 核心路径

```
memory_engine.py : recall_async()
    │
    ├─ 认证租户
    ├─ 过滤 deprecated "opinion" 类型
    ├─ 可选扩展钩子（validate_recall / on_recall_complete）
    ├─ Budget 映射: LOW=100, MID=300, HIGH=1000（节点预算）
    │
    ├─ _search_semaphore（背压控制，防止 DB 过载）
    │
    └─ _search_with_retries()（指数退避重试，最多 3 次）
            │
            └─ search/retrieval.py : retrieve_all_fact_types_parallel()
                    │
                    ├─ 生成 query embedding（单次调用，所有 fact_type 共用）
                    ├─ 分析时间约束（QueryAnalyzer）
                    │
                    └─ 四路并行（asyncio.gather）：
                         ├─ semantic + BM25 (合并单 SQL 查询)
                         ├─ graph retrieval (BFS/MPFP 传播)
                         └─ temporal retrieval (如检测到时间约束)
                                │
                    ┌──────────┘
                    ▼
            fusion.py : reciprocal_rank_fusion()   # RRF 合并
                    │
                    ▼
            reranking.py : CrossEncoderReranker.rerank()  # 神经重排
                    │
                    ▼
            apply_combined_scoring()  # 时间衰减打分
                    │
                    ▼
            MMR diversity filter  # 最大边际相关性去重
                    │
                    ▼
            Token budget filter   # 控制 max_tokens
```

---

## 5. DFS：recall 关键机制深挖

### 5.1 四路并行检索

**文件：** `engine/search/retrieval.py`

#### 语义检索（Semantic）

```python
# retrieve_semantic_bm25_combined() 中的 HNSW 查询
# 每个 fact_type 一个独立子查询，允许 planner 使用 partial HNSW index

# 关键参数：
# hnsw_fetch = max(limit * 5, 100)  # 过采样 5x 补偿 HNSW 近似误差
# similarity >= 0.3                 # 相似度阈值
# ef_search = 200                   # 连接时设置，提升稀疏图 recall

sem_arm = f"""
(SELECT id, text, ...,
        1 - (embedding <=> $1::vector) AS similarity,
        'semantic' AS source
 FROM {table}
 WHERE bank_id = $2
   AND fact_type = '{ft}'
   AND (1 - (embedding <=> $1::vector)) >= 0.3
 ORDER BY embedding <=> $1::vector
 LIMIT {hnsw_fetch})
"""
```

#### BM25 全文检索

```python
# 支持三种实现（配置驱动）：
# 1. vchord:        search_vector <&> to_bm25query(...)  ← 最准确
# 2. pg_textsearch: text <@> to_bm25query(...)
# 3. native:        ts_rank_cd(search_vector, to_tsquery(...))  ← 默认

# query 预处理：去标点、lowercase、split
tokens = re.sub(r"[^\w\s]", " ", query_text.lower()).split()
```

**为什么 Semantic + BM25 合并在单个 SQL？**

避免两次数据库往返，利用 PostgreSQL 的 UNION ALL 查询优化器，让每个 arm 独立使用自己的 partial index。

#### Graph 检索（BFS Spreading Activation）

**文件：** `engine/search/graph_retrieval.py`

```python
class BFSGraphRetriever:
    entry_point_limit    = 5     # 从语义结果中选取入口节点
    entry_point_threshold = 0.5  # 语义相似度门槛
    activation_decay     = 0.8   # 每跳衰减（0.8^hop）
    min_activation       = 0.1   # 停止扩散的最小激活值
    batch_size           = 20    # 邻居批量获取

# 算法流程：
# 1. 从 semantic 结果中选取 top-5 高相似度节点作为入口
# 2. BFS 展开：visited_nodes + activation_score 字典
# 3. 每跳激活值 *= activation_decay
# 4. 激活值 < min_activation 时停止
# 5. 因果边额外 2.0x boost，enables/prevents 边 1.5x boost
```

这模仿的是**神经网络的激活传播**（Spreading Activation），来自认知科学的联想记忆模型。

#### 时间检索（Temporal）

```sql
-- retrieve_temporal_combined() 两阶段优化：
-- Phase 1: date_ranked — 只用日期索引过滤，不算向量距离
-- Phase 2: sim_ranked  — 对 top-50 候选计算 embedding 距离

WITH date_ranked AS MATERIALIZED (
    SELECT id, fact_type,
           ROW_NUMBER() OVER (PARTITION BY fact_type
               ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC) AS rn
    FROM memory_units
    WHERE bank_id = $2
      AND fact_type = ANY($3)
      AND (occurred_start BETWEEN $4 AND $5 OR mentioned_at BETWEEN $4 AND $5 ...)
),
sim_ranked AS (
    SELECT mu.*, 1 - (mu.embedding <=> $1::vector) AS similarity,
           ROW_NUMBER() OVER (PARTITION BY mu.fact_type ORDER BY mu.embedding <=> $1::vector) AS sim_rn
    FROM date_ranked dr JOIN memory_units mu ON mu.id = dr.id
    WHERE dr.rn <= 50 AND similarity >= $6
)
SELECT * FROM sim_ranked WHERE sim_rn <= 10
```

**时间区间由 QueryAnalyzer 提取**（dateparser / transformer 两种模式），将 "去年夏天" 这类自然语言解析为具体日期区间。

---

### 5.2 Reciprocal Rank Fusion (RRF)

**文件：** `engine/search/fusion.py`

```python
def reciprocal_rank_fusion(result_lists: list[list[RetrievalResult]], k: int = 60):
    """
    RRF 公式：score(d) = Σ 1/(k + rank(d))
    k=60 是 RRF 原论文的推荐超参数，在稀疏/密集检索中都表现稳定。
    """
    for source_idx, results in enumerate(result_lists):
        for rank, retrieval in enumerate(results, start=1):
            doc_id = retrieval.id
            rrf_scores[doc_id] += 1.0 / (k + rank)
            source_ranks[doc_id][f"{source_name}_rank"] = rank

    # 按 rrf_score 降序排列
    return sorted(merged_results, key=lambda x: x.rrf_score, reverse=True)
```

**四路结果列表顺序**（映射到 source_names）：
```python
source_names = ["semantic", "bm25", "graph", "temporal"]
```

RRF 的核心优势：**不依赖各路检索分数的绝对值**，只依赖排名（rank），天然解决了不同检索方法分数量纲不一致的问题。

---

### 5.3 Cross-Encoder 神经重排

**文件：** `engine/search/reranking.py`

```python
class CrossEncoderReranker:
    async def rerank(self, query: str, candidates: list[MergedCandidate]):
        # 构造 (query, document) 对，加入日期信息增强时间感知
        for candidate in candidates:
            doc_text = candidate.retrieval.text
            if candidate.retrieval.context:
                doc_text = f"{candidate.retrieval.context}: {doc_text}"
            if candidate.retrieval.occurred_start:
                date_readable = occurred_start.strftime("%B %d, %Y")
                doc_text = f"[Date: {date_readable}] {doc_text}"
            pairs.append([query, doc_text])

        # 调用 cross-encoder（本地: ms-marco-MiniLM-L-6-v2 / 远程: TEI）
        scores = await self.cross_encoder.predict(pairs)

        # logits → sigmoid → [0, 1]
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        normalized_scores = [sigmoid(s) for s in scores]
```

**为什么用 Cross-Encoder 而不是 Bi-Encoder？**

- **Bi-Encoder**（向量检索用）：query 和 document 分别编码，效率高但精度有限
- **Cross-Encoder**（重排用）：query + document 拼接后联合编码，能捕捉细粒度交互，精度更高
- 代价：时间复杂度 O(N) vs Bi-Encoder 的 O(1)（查询时），所以只对 top-K 候选重排

---

### 5.4 Combined Scoring — 时间感知打分

**文件：** `engine/search/reranking.py : apply_combined_scoring()`

```python
# 最终打分公式（乘法 boost，保证比例性）：
recency_boost  = 1 + 0.2 * (recency  - 0.5)   # ±10%
temporal_boost = 1 + 0.2 * (temporal - 0.5)   # ±10%
combined_score = cross_encoder_score_normalized * recency_boost * temporal_boost

# recency 计算（线性衰减，365天归一化）：
days_ago = (now - occurred_start).total_seconds() / 86400
recency  = max(0.1, min(1.0, 1.0 - days_ago / 365))
# 结果：今天=1.0, 6个月前≈0.5, 1年前≈0.1

# temporal_proximity：由时间检索路径设置（与查询时间区间的匹配程度），
#                    非时间查询默认 0.5（中性，boost 因子 = 1.0）
```

**设计精髓**：加法 boost 会让低相关性的近期文档"虚假升权"，而乘法 boost 确保 **boost 效果与基础相关性成正比**——强相关的结果受时间因素影响更大，弱相关的结果受影响更小。

---

## 6. 真实运行示例

### 示例一：Retain

**请求：**
```bash
curl -X POST http://localhost:8000/v1/default/banks/user_alice/memories/retain \
  -H "Content-Type: application/json" \
  -d '{
    "items": [{
      "content": "Alice 昨晚和她的室友 Sarah 一起看了《Oppenheimer》，她觉得剧情很震撼，特别是三位一体测试那段。",
      "timestamp": "2024-03-15T22:00:00",
      "document_id": "chat_session_001"
    }]
  }'
```

**内部执行（关键日志输出）：**
```
RETAIN_BATCH START: user_alice
Batch size: 1 content items, 52 chars
========================================
  Extract facts: 2 facts, 1 chunks from 1 contents in 1.23s
  Generate embeddings: 2 embeddings in 0.18s
  Document tracking: 1 documents in 0.01s
  Insert facts: 2 units in 0.02s
  Process entities: 3 links in 0.05s
  Temporal links: 1 links in 0.01s
  Semantic links: 0 links in 0.01s
  Entity links: 3 links in 0.01s
  Causal links: 0 links in 0.01s
RETAIN_BATCH DONE in 1.52s
```

**LLM 提取结果（2 个 fact）：**

Fact 1:
```json
{
  "what": "Alice 和室友 Sarah 一起观看了电影《Oppenheimer》",
  "when": "2024-03-15 晚上",
  "where": "N/A",
  "who": "Alice（用户）; Sarah（Alice 的室友）",
  "why": "Alice 感到震撼，对三位一体测试场景印象深刻",
  "fact_type": "experience",
  "occurred_start": "2024-03-15T22:00:00",
  "entities": [{"text": "Alice"}, {"text": "Sarah"}, {"text": "Oppenheimer"}]
}
```

Fact 2:
```json
{
  "what": "《Oppenheimer》的三位一体核试验场景令人震撼",
  "when": "N/A",
  "where": "N/A",
  "who": "N/A",
  "why": "Alice 特别提到了这个场景，说明她对此印象深刻",
  "fact_type": "world",
  "entities": [{"text": "Oppenheimer"}, {"text": "三位一体测试"}]
}
```

**响应：**
```json
{
  "success": true,
  "bank_id": "user_alice",
  "items_count": 1,
  "async": false,
  "usage": {"input_tokens": 312, "output_tokens": 187, "total_tokens": 499}
}
```

---

### 示例二：Recall

**请求：**
```bash
curl -X POST http://localhost:8000/v1/default/banks/user_alice/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Alice 和她朋友看过哪些电影？",
    "budget": "mid",
    "max_tokens": 2048,
    "types": ["world", "experience"]
  }'
```

**内部执行流程（关键日志）：**
```
[RECALL user_ali] Starting recall for query: Alice 和她朋友看过哪些电影？...
[RECALL] Query embedding generated in 0.08s
[RECALL] QueryAnalyzer: no temporal constraint detected
[RECALL] Parallel retrieval start (world, experience)
  semantic:  12 results in 0.045s (world=6, experience=6)
  bm25:       8 results in 0.045s (world=4, experience=4)
  graph:     15 results in 0.120s (BFS, 3 entry points, 4 hops)
  temporal:  skipped (no date constraint)
[RECALL] RRF merge: 28 unique candidates
[RECALL] Cross-encoder rerank: 28 pairs in 0.234s
[RECALL] Combined scoring (recency + temporal)
[RECALL] Token filter: 3 facts selected, 412 tokens
```

**响应（节选）：**
```json
{
  "results": [
    {
      "id": "unit_abc123",
      "text": "Alice 和室友 Sarah 一起观看了电影《Oppenheimer》 | Involving: Alice（用户）; Sarah（Alice 的室友）| Alice 感到震撼，对三位一体测试场景印象深刻",
      "type": "experience",
      "entities": ["Alice", "Sarah", "Oppenheimer"],
      "occurred_start": "2024-03-15T22:00:00Z",
      "occurred_end": null,
      "mentioned_at": "2024-03-15T22:00:00Z"
    }
  ]
}
```

---

## 7. 补充知识：论文/设计思想映射

### 7.1 5W 事实提取 ← 结构化知识表示

- **灵感来源**：新闻信息提取（Journalism 5W）、知识图谱三元组（SPO）
- **工程意义**：比原始文本向量更密集、噪音更少，检索时语义更集中
- **类似工作**：KGQA（知识图谱问答），GraphRAG 的实体提取

### 7.2 四路并行检索 ← Hybrid Search

- **论文**：[Sparse, Dense, and Attentional Representations for Text Retrieval (2021)](https://arxiv.org/abs/2005.00181)
- **BEIR Benchmark** 显示混合检索在多数任务上优于单一方法
- **Hindsight 扩展**：在 semantic + BM25 基础上加入 graph（关联性）和 temporal（时序性）

### 7.3 BFS Spreading Activation ← 认知科学 + 图神经网络

- **经典来源**：Collins & Loftus (1975) Spreading Activation Theory
- **工程实现类比**：PersonalizedPageRank（PPR），GraphSAGE 邻居聚合
- **激活衰减**：`activation *= 0.8^hop` 类似 PPR 中的 `(1-α)*A^t` 矩阵乘法

### 7.4 Reciprocal Rank Fusion (RRF)

- **原论文**：[Cormack, Clarke & Buettcher, SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **k=60** 是原论文中的经验值，对 top-1000 结果效果最稳定
- **优势**：分数无关（rank-based），天然解决多路检索分数量纲不统一问题

### 7.5 Cross-Encoder 重排 ← 两阶段检索

- **论文**：[MS-MARCO](https://microsoft.github.io/msmarco/) + [ColBERT](https://arxiv.org/abs/2004.12832) 推广的两阶段范式
- **默认模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2`（MiniLM 蒸馏，精度/速度平衡）
- **Sigmoid 归一化**：cross-encoder 输出 raw logits，sigmoid 映射到 [0,1] 方便后续组合

### 7.6 Delta Retain ← Incremental Processing

- **工程模式**：类似 incremental compilation（only recompile changed files）
- **在 AI 系统中的对应**：GraphRAG 的增量知识图谱更新，RAG 的文档版本管理

### 7.7 Bank 隔离 ← Multi-Tenancy

- **实现**：PostgreSQL schema-per-tenant（Context Var `_current_schema`）
- **fq_table()** 自动注入 schema 前缀，所有查询都隔离在租户 schema 内
- **好处**：数据隔离彻底，迁移/备份/删除都是 schema 级操作

---

## 8. 性能与容错设计

### 8.1 背压控制

```python
# memory_engine.py
self._search_semaphore = asyncio.Semaphore(max_concurrent_searches)
# 防止大量并发 recall 请求压垮 PostgreSQL 连接池
```

### 8.2 连接池重试（指数退避）

```python
# recall_async() 中：
for attempt in range(max_retries + 1):  # max_retries = 3
    try:
        result = await self._search_with_retries(...)
        break
    except asyncpg.TooManyConnectionsError:
        wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
        await asyncio.sleep(wait_time)
```

### 8.3 数据库事务 + 死锁重试

```python
# orchestrator.py
async def _run_db_work():
    async with acquire_with_retry(pool) as conn:
        async with conn.transaction():
            # 所有写操作在单事务中
            ...

# retry_with_backoff 自动重试死锁（PostgreSQL 40P01）
```

### 8.4 Batch API 崩溃恢复

```python
# 使用 OpenAI/Groq Batch API 时，batch_id 存入 operation metadata
# 服务器重启后可通过 batch_id 重新轮询结果，不需要重新提交
```

### 8.5 HNSW ef_search 全局设置

```python
# memory_engine.py 连接池初始化时：
# SET hnsw.ef_search = 200
# 提升稀疏图（cold start）场景下的 recall，代价是轻微的延迟增加
```

### 8.6 Token Budget 软限制

```python
# recall_async() 返回阶段：
# 按 combined_score 降序遍历，累计 token 数
# 当下一条结果会导致超出 max_tokens 时停止
# 已经返回的结果不会被截断
# 效果：max_tokens=2048 → 返回 5-20 条 fact（取决于每条长度）
```

---

## 总结：设计哲学对比

| 维度 | 朴素 RAG | Hindsight |
|------|---------|-----------|
| 存储单元 | 原始 chunk | 结构化 5W Fact |
| 检索策略 | 单路向量检索 | 四路并行（向量+BM25+图+时间） |
| 合并策略 | 无 / 简单分数 | Reciprocal Rank Fusion |
| 精度优化 | 无 | Cross-Encoder 神经重排 |
| 时间感知 | 无 | 时间检索 + recency decay |
| 增量更新 | 全量重建 | Delta Retain（chunk hash 比较） |
| 实体关系 | 无 | Entity Resolver + 异构图 |
| 多租户 | 无 / 应用层过滤 | PostgreSQL schema 隔离 |

**核心设计思路**：Hindsight 把"记忆"类比成人类的认知系统——不是原始录音，而是经过提炼的**结构化语义网络**，检索时模拟**联想激活**而非单纯的向量查找。
