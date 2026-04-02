# `_pre_resolve_phase1` 原理精讲

## 1. 这个阶段在整个 retain 流程里的位置

`_pre_resolve_phase1` 定义在 [orchestrator.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/orchestrator.py)，它位于 retain 主流程的中间层：

1. 前面阶段已经完成了事实抽取和 embedding 生成。
2. 然后进入 `_pre_resolve_phase1`，先做一批“昂贵但主要是读”的预处理。
3. 接着才进入 `_insert_facts_and_links`，在真正的数据库事务里写入 `memory_units`、`unit_entities`、`temporal/semantic/causal links`。
4. 最后 Phase 3 再补 `entity_links` 这类更偏展示的数据。

也就是说，它不是“抽事实”的阶段，也不是“真正落库”的阶段，而是一个事务前预计算阶段。

它的核心目标只有一个：

**把后面写事务里本来需要做的慢读操作，提前搬到事务外做完。**

---

## 2. 它到底解决了什么问题

如果没有 `_pre_resolve_phase1`，那 retain 在写数据库事务时不仅要：

- 插入 facts
- 插入 unit_entities
- 插入各种 links

还要同时做两类很重的查询：

- 实体归一化：把 `"OpenAI"`、`"Open AI"`、`"openai"` 这种名字匹配到库里的同一个 entity
- 语义 ANN 检索：拿新 facts 的 embedding 去向量索引里找语义近邻

这两件事都偏“读”：

- 实体归一化会触发 trigram / 共现 / 打分
- ANN 会走向量索引近邻查询

这些查询本身不一定改很多数据，但它们慢，而且如果放在事务里做，事务会持锁更久。并发 retain 多了以后，就容易出现：

- 事务时间长
- 行锁持有时间长
- 超时
- 吞吐下降

所以 `_pre_resolve_phase1` 的设计思想是：

**把“先查清楚要写什么”这件事提前做掉，事务里只保留“真正写进去”这件事。**

---

## 3. 这个阶段实际做了哪两件事

`_pre_resolve_phase1` 只做两类预计算：

1. 实体解析（entity resolution）
2. 语义近邻预查询（semantic ANN）

它不做 facts 插入，不做 `unit_entities` 插入，也不直接写 `entity_links`。

### 3.1 实体解析

这一步的目标是：

**把每条新 fact 里提到的实体，提前映射成数据库中的 canonical entity id。**

调用链是：

- `_pre_resolve_phase1`
- `entity_processing.resolve_entities(...)`
- `link_utils.resolve_entities_only(...)`
- `entity_resolver.resolve_entities_batch(...)`

它做的不是“简单字符串去重”，而是把实体变成后续可写入、可关联的稳定 ID。

### 3.2 语义 ANN 预查询

这一步的目标是：

**先根据每条 fact 的 embedding，去找库里语义上接近的已有 unit。**

调用的是：

- `compute_semantic_links_ann(...)`

返回的是一组候选 semantic links，不是最终事务提交后的完整图结构。

它本质上是在事务前说：

“这批新 facts 以后大概率会和哪些旧 memory units 建 semantic link，我先查出来。”

---

## 4. 这个函数内部是怎么做的

下面按数据流解释，不讲“怎么调用”，只讲“它内部怎么运转”。

### 4.1 先收集用户显式提供的实体

函数先构造：

```python
user_entities_per_content = {idx: content.entities for idx, content in enumerate(contents) if content.entities}
```

这一步不是多余的。原因是系统里的实体来源不只一种：

- 一部分来自 LLM 从 fact 中抽出来的实体
- 一部分可能是用户在输入 content 时显式带进来的实体

后面实体解析阶段会把这两部分合并。

#### 例子

假设用户传入一段内容：

```text
昨天和 Sam Altman 聊了 GPT-5.4 的 API 定价。
```

同时用户自己还附带了结构化实体：

```json
[
  {"text": "Sam Altman", "type": "PERSON"},
  {"text": "GPT-5.4", "type": "PRODUCT"}
]
```

而 LLM 抽取 facts 时可能只抽到了：

- `Sam Altman`
- `API`

那么 `_pre_resolve_phase1` 不会只信 LLM 的实体，而是会把用户实体也并进后续解析链路。

这意味着最终参与解析的实体集合更完整。

---

### 4.2 构造占位 `unit_id`

函数里有这一步：

```python
placeholder_unit_ids = [str(i) for i in range(len(processed_facts))]
```

为什么要这么做？

因为此时新的 `memory_units` 还没有真正插入数据库，所以根本没有真实 UUID。

但后面的实体解析和 semantic ANN 结果，又都需要知道：

- “这个解析出来的 entity 属于哪条 fact”
- “这条 semantic link 的 from_unit_id 是谁”

所以这里先用一个临时编号顶上，比如：

- 第 0 条 fact -> `"0"`
- 第 1 条 fact -> `"1"`
- 第 2 条 fact -> `"2"`

这些占位 ID 的作用不是持久化，而是：

**在事务前，把各种预计算结果先挂在一个可追踪的本批次标识上。**

#### 例子

如果本批次抽出了 3 条 facts：

1. `OpenAI released a new model`
2. `Sam Altman discussed pricing`
3. `The API cost increased`

那这里会先生成：

```text
placeholder_unit_ids = ["0", "1", "2"]
```

后面无论是实体解析结果还是 ANN 结果，都先写成“和 `"0"` / `"1"` / `"2"` 相关”。

等真正入库得到 UUID 后，再统一 remap。

这就是为什么后面还有 `_remap_phase1_results(...)`。

---

### 4.3 提前拿出 embeddings

函数会做：

```python
embeddings = [fact.embedding for fact in processed_facts]
```

这里不是重新生成 embedding，只是把前面步骤已经算好的向量拿出来。

原因很直接：

- semantic ANN 依赖 embedding
- `_pre_resolve_phase1` 正好就是负责做 semantic ANN 预查询

所以它复用前面已经算好的向量，不重复计算。

---

### 4.4 用独立连接执行 Phase 1

核心设计点是：

```python
async with acquire_with_retry(pool) as resolve_conn:
```

它显式拿了一个单独的数据库连接，目的是把这阶段和主写事务隔开。

注意，这里不是单纯“风格上分层”，而是一个并发控制策略：

- Phase 1 用单独连接做慢读
- Phase 2 再用事务连接做写

这样后面的事务不会在持锁期间再去做大范围模糊匹配和近邻检索。

#### 例子

想象一个 bank 里已经有 100 万条 memory units。

新来一批 facts 时：

- 实体解析可能要扫候选 entity
- ANN 可能要查很多近邻候选

如果这些查询放在事务里，事务就会在“已经开始写，但还没写完”的状态里停留很久。

而现在的做法是：

- 先在事务外把候选结果找出来
- 真到事务里只执行确定的插入和关联

这会大幅缩短事务的关键路径。

---

### 4.5 先做实体解析

在独立连接里，函数先调用：

```python
entity_processing.resolve_entities(...)
```

这一步的结果有三份：

- `resolved_entity_ids`
- `entity_to_unit`
- `unit_to_entity_ids`

这三份结果分别代表不同层次的信息。

#### `resolved_entity_ids`

它是一个扁平列表，顺序对应“展开后的实体列表”。

例如有两条 facts：

- fact `"0"` 有实体：`["OpenAI", "Sam Altman"]`
- fact `"1"` 有实体：`["GPT-5.4"]`

展开后变成 3 个待解析实体：

1. `"OpenAI"`
2. `"Sam Altman"`
3. `"GPT-5.4"`

那么 `resolved_entity_ids` 可能就是：

```text
["e_101", "e_205", "e_390"]
```

#### `entity_to_unit`

它记录这个扁平实体列表里的每一个元素，原本属于哪个占位 unit。

比如：

```text
[
  ("0", 0, 2026-04-03T10:00:00Z),
  ("0", 1, 2026-04-03T10:00:00Z),
  ("1", 0, 2026-04-03T10:05:00Z)
]
```

含义是：

- 第一个解析实体属于占位 unit `"0"`，是这个 unit 里的第 0 个实体
- 第二个也属于 `"0"`
- 第三个属于 `"1"`

#### `unit_to_entity_ids`

它是反向聚合结果，直接告诉你每个占位 unit 关联了哪些 entity id。

例如：

```python
{
  "0": ["e_101", "e_205"],
  "1": ["e_390"]
}
```

这个结构在后面写 `unit_entities` 时非常有用，因为它已经非常接近最终要落库的关系。

---

## 5. 实体解析内部为什么要在这里做

实体解析阶段最重要的不是“拿到名字”，而是“拿到稳定的 canonical entity id”。

因为后面真正落库时，`unit_entities` 写的是：

- `unit_id`
- `entity_id`

不是原始文本。

也就是说，后面写事务不希望再临时判断：

- `"OpenAI"` 和 `"open ai"` 是不是同一个实体
- `"Sam"` 是否应该归并到 `"Sam Altman"`
- 如果库里没有这个实体，要不要创建新 entity

这些决策如果放在事务里做，既慢又复杂。

`_pre_resolve_phase1` 的意义就是：

**在进入事务之前，把“实体名”变成“实体 ID”。**

这样 Phase 2 里只需要：

- 拿到真实 `unit_id`
- 把 `(unit_id, entity_id)` 批量插入

---

## 6. 再做 semantic ANN 预查询

实体解析之后，函数会准备 `semantic_ann_links = []`，然后在允许的情况下调用：

```python
compute_semantic_links_ann(...)
```

它的输入本质上是：

- 当前 bank
- 当前批次每条新 fact 的占位 `unit_id`
- 每条 fact 的 embedding
- 每条 fact 的 `fact_type`

它的输出是一批“候选语义 links”，形状大致类似：

```text
[
  (from_placeholder_unit_id, to_existing_unit_id, "semantic", weight, None),
  ...
]
```

这里最关键的是：

- `from` 端还是占位 `unit_id`
- `to` 端已经是库里真实存在的旧 unit

#### 例子

假设新 fact `"0"` 的内容是：

```text
OpenAI cut the price of its latest API model.
```

库里已经有两个旧 unit：

- `u_9001`: `OpenAI reduced API pricing in March`
- `u_7310`: `Anthropic released a new Claude version`

ANN 预查询可能返回：

```text
[
  ("0", "u_9001", "semantic", 0.92, None)
]
```

意思是：

- 本批次第 `"0"` 条新 fact
- 语义上和已有的 `u_9001` 很近
- 后面大概率要建一个 semantic link

这个结果此时还不能直接落库，因为 `"0"` 不是最终 `memory_units.id`。

所以它先暂存，等 Phase 2 插入新 facts 得到真实 UUID 后再 remap。

---

## 7. 为什么 streaming 模式里可能跳过 semantic ANN

函数里有个参数：

```python
skip_semantic_ann: bool = False
```

在 streaming retain 场景里，调用方会传 `True`，原因不是“语义 link 不重要”，而是：

**逐小批次做 ANN，会让后续批次越来越慢。**

原因是：

- 每来一小批都要查一次整个 bank 的向量近邻
- bank 越大，这个查询成本越明显
- 小批多轮时，整体开销会被放大

所以 streaming 模式常见策略是：

- Phase 1 只做实体解析
- semantic ANN 留到后面统一做一次

这是一种吞吐优化，而不是功能删除。

---

## 8. 这个阶段为什么一定要返回结构化结果，而不是直接写库

`_pre_resolve_phase1` 最后返回的是：

```python
Phase1Result(
    entities=EntityResolutionResult(...),
    semantic_ann_links=semantic_ann_links,
)
```

它返回“结果对象”，而不是自己就把所有东西写到库里，原因有两个。

### 8.1 它此时还没有真实 `unit_id`

不管是 `unit_entities` 还是 semantic links，都需要新的 `memory_units.id`。

但 Phase 1 发生在插入 facts 之前，所以它只能先返回：

- 针对占位 `unit_id` 的实体结果
- 针对占位 `unit_id` 的 ANN 结果

等 Phase 2 完成 `insert_facts_batch(...)` 之后，才能 remap 到真实 UUID。

### 8.2 它的职责是“准备写入信息”，不是“自己提交”

这里故意把职责切开：

- Phase 1：准备数据
- Phase 2：事务提交
- Phase 3：补展示数据

这让每个阶段的性能目标和失败语义都更清晰。

---

## 9. 它在整个流程中的真正作用

如果只用一句话概括 `_pre_resolve_phase1` 的作用，可以这样说：

**它是 retain 流程中的“事务前预计算缓冲层”。**

更具体一点，它起了 4 个关键作用：

### 9.1 缩短事务关键路径

把最慢的读操作提前做掉，让事务阶段只剩下确定性的写。

### 9.2 把“文本级实体”变成“ID 级实体”

后续数据库写入不再依赖模糊文本判断，而是依赖稳定的 `entity_id`。

### 9.3 把未来要建的 semantic links 先算出来

这样 Phase 2 不用一边持锁一边查向量近邻。

### 9.4 建立从“本批次临时 fact”到“最终数据库记录”的桥

它先用占位 `unit_id` 把关系组织好，后面只需要一次 remap 就能接上真实 UUID。

---

## 10. 用一个完整例子串起来

假设有一条输入内容：

```text
On April 3, Sam Altman said OpenAI would lower GPT-5.4 API prices.
```

前面步骤已经做完：

- fact extraction 得到一条 fact
- embedding generation 得到一个向量

现在进入 `_pre_resolve_phase1`。

### 第一步：构造占位 unit

因为现在还没插入 `memory_units`，所以先给这条 fact 一个临时 ID：

```text
placeholder_unit_ids = ["0"]
```

### 第二步：实体解析

从这条 fact 中拿到实体，比如：

- `Sam Altman`
- `OpenAI`
- `GPT-5.4`

然后 `entity_resolver.resolve_entities_batch(...)` 可能返回：

```text
resolved_entity_ids = ["e_person_12", "e_org_8", "e_product_77"]
```

并形成：

```python
unit_to_entity_ids = {
  "0": ["e_person_12", "e_org_8", "e_product_77"]
}
```

### 第三步：semantic ANN

拿这条 fact 的 embedding 去查 bank 中已有的 units，找到一个很相近的旧 unit：

```text
u_old_301 = "OpenAI discussed API pricing changes last month"
```

于是 ANN 结果可能是：

```text
[
  ("0", "u_old_301", "semantic", 0.91, None)
]
```

### 第四步：返回给 Phase 2

Phase 1 现在不写库，只把这些结构化结果返回。

### 第五步：Phase 2 真正插入 facts

假设插入后，数据库给这条新 fact 分配了真实 UUID：

```text
u_new_999
```

这时系统会把所有 Phase 1 结果中的 `"0"` remap 成 `u_new_999`：

- `("0", "u_old_301", "semantic", 0.91, None)`  
  变成  
  `("u_new_999", "u_old_301", "semantic", 0.91, None)`

- `"0" -> ["e_person_12", "e_org_8", "e_product_77"]`  
  变成  
  `"u_new_999" -> ["e_person_12", "e_org_8", "e_product_77"]`

这样事务里就可以直接写：

- `memory_units: u_new_999`
- `unit_entities: (u_new_999, e_person_12)`, `(u_new_999, e_org_8)`, `(u_new_999, e_product_77)`
- `memory_links: (u_new_999 -> u_old_301, semantic, 0.91)`

整个过程中，最慢的“实体归一化”和“ANN 查邻居”都已经在事务外完成了。

---

## 11. 最后一句话总结

`_pre_resolve_phase1` 的本质不是“做点预处理”，而是：

**先把本批次新 facts 变成一组已经准备好落库的“关系草图”，只是其中的新 unit 还暂时用占位 ID 表示。**

等 Phase 2 拿到真实 UUID，这张“关系草图”就能被快速、原子地写进数据库。
