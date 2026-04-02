# `_build_extraction_prompt_and_schema()` 说明

文件位置：
[fact_extraction.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py)

这份说明专门解释 `_build_extraction_prompt_and_schema(config)` 做了什么，为什么它看起来有很多分支，以及它和 `retain_extraction_mode="chunks"` 的关系。

## 先说结论

你看到的直觉“最常见的 `extraction_mode` 可能是 `chunks`”是合理的，但要注意：

- `chunks` 模式根本不会进入 `_build_extraction_prompt_and_schema()`
- 这个函数只在“需要调用 LLM 提取事实”的路径里才会被用到
- 所以它处理的其实是这些模式：
  - `concise` 或默认分支
  - `custom`
  - `verbose`
  - `verbatim`

## 调用链路

最外层入口是：

- [fact_extraction.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py#L2197)

在 `extract_facts_from_contents()` 里，最先判断的是：

```python
if config.retain_extraction_mode == "chunks":
    return _extract_facts_chunks(contents, config)
```

也就是说：

1. 如果 mode 是 `chunks`
2. 系统直接把每个 chunk 当成一个 memory unit
3. 不走 LLM
4. 不需要 prompt
5. 当然也不会调用 `_build_extraction_prompt_and_schema()`

只有在非 `chunks` 模式下，才会继续走 `extract_facts_from_text()` -> `_extract_facts_with_auto_split()` -> `_extract_facts_from_chunk()`。

而 `_extract_facts_from_chunk()` 里才会调用：

```python
prompt, response_schema = _build_extraction_prompt_and_schema(config)
```

## 这个函数的职责

`_build_extraction_prompt_and_schema(config)` 只做两件事：

1. 构造最终发给 LLM 的 `system prompt`
2. 构造与这个 prompt 配套的 `response_schema`

它不做这些事：

- 不切 chunk
- 不调用 LLM
- 不解析 LLM 返回
- 不做事实后处理
- 不决定是否进入 `chunks` 模式

所以它本质上是一个“提示词与输出 schema 装配器”。

## 为什么会有很多分支

因为这个函数要同时解决三个维度的变化：

1. 提取风格不同
2. 输出字段不同
3. label 配置可能动态变化

这三个维度叠加后，分支自然就多了。

---

## 第一层分支：`retain_mission`

代码位置大致在：

- [fact_extraction.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py#L825)

这段：

```python
retain_mission = getattr(config, "retain_mission", None)
if retain_mission:
    retain_mission_section = ...
else:
    retain_mission_section = ""
```

它的作用不是选择模式，而是给 prompt 加一个“保留重点”前缀。

如果 bank 配了 `retain_mission`，模型会先看到类似：

```text
FOCUS — What to retain for this bank
...
```

然后才进入后面的正式提取规则。

所以这层分支的意义是：

- 有 `retain_mission`：prompt 带偏好说明
- 没有 `retain_mission`：prompt 不带这段

这不改变 schema，只影响提示词内容。

## 第二层分支：按 `extraction_mode` 选基础 prompt

代码位置大致在：

- [fact_extraction.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py#L837)

这里是最核心的一组分支。

### 1. `custom`

逻辑：

- 如果 `retain_custom_instructions` 为空，退回 `CONCISE_FACT_EXTRACTION_PROMPT`
- 如果不为空，使用 `CUSTOM_FACT_EXTRACTION_PROMPT`

原因：

- `custom` 只是允许你插入自定义指令
- 但如果你根本没提供自定义内容，系统只能回到 concise 默认模板

### 2. `verbose`

使用：

- `VERBOSE_FACT_EXTRACTION_PROMPT`

特点：

- 鼓励模型尽量详细保留信息
- 输出通常更长
- 适合你想尽量保留细节的场景

### 3. `verbatim`

使用：

- `VERBATIM_FACT_EXTRACTION_PROMPT`

特点：

- 不依赖模型生成最终 `fact_text`
- 模型主要提取元数据，比如时间、人物、实体
- 最终 `fact_text` 会在后面被原始 chunk 文本回填

### 4. 默认分支

如果不是 `custom` / `verbose` / `verbatim`，就走：

- `CONCISE_FACT_EXTRACTION_PROMPT`

也就是说，普通 LLM 提取模式的默认心智模型可以理解为：

- “简洁提取”

## 第三层分支：是否启用 causal links

代码位置大致在：

- [fact_extraction.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py#L864)

这一层不仅影响 prompt，也影响 schema。

### `verbatim` 特殊处理

如果 mode 是 `verbatim`，这里直接固定为：

- `base_fact_class = VerbatimExtractedFact`
- `base_response_class = VerbatimFactExtractionResponse`

原因很直接：

- `verbatim` 模式不是让模型自由生成 fact 文本
- 因果关系依赖的是“fact 语义单元之间的关系”
- 但 `verbatim` 最终保存的是原始 chunk 文本
- 所以这里不启用 causal relation 设计

### `extract_causal_links = True`

如果不是 `verbatim`，且开启了 `retain_extract_causal_links`：

- prompt 追加 `CAUSAL_RELATIONSHIPS_SECTION`
- schema 改成带 `causal_relations` 的版本

具体又会按 verbose / 非 verbose 再细分：

- `verbose` -> `ExtractedFactVerbose` / `FactExtractionResponseVerbose`
- 非 `verbose` -> `ExtractedFact` / `FactExtractionResponse`

### `extract_causal_links = False`

如果关闭因果关系提取：

- 使用不带 `causal_relations` 的 schema
- 即 `ExtractedFactNoCausal` / `FactExtractionResponseNoCausal`

这样做的价值是：

- prompt 更简单
- 输出 schema 更短
- 降低模型出错概率

## 第四层分支：是否配置 entity labels

代码位置大致在：

- [fact_extraction.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py#L877)

这是这个函数里最容易看花的部分，因为它不是简单选一个常量，而是“动态造 schema”。

### 先做什么

先读取配置：

- `entity_labels`
- `entities_allow_free_form`

然后调用：

```python
labels_section = _build_labels_prompt_section(labels_cfg, free_form_entities)
```

如果返回非空，就把这段追加到 prompt 后面。

所以这一层先影响 prompt：

- 告诉模型有哪些 label group
- 每个 group 能填什么
- 是单值、多值还是自由文本

### 再做什么

然后它看是否真的配置了 label attributes：

```python
if labels_cfg and labels_cfg.attributes:
```

如果有，就进一步：

1. 用 `build_labels_model(labels_cfg)` 动态生成一个 `LabelsModel`
2. 把这个 `LabelsModel` 作为 `labels` 字段加进 fact schema
3. 如有需要，还把 `entities` 改成“保留字段但要求为空”
4. 再用 `create_model(...)` 动态生成：
   - `DynamicFact`
   - `DynamicResponse`

这就是为什么你会看到这里不像前面那样只是选一个类，而是要现造模型。

## 为什么要动态造 schema

因为 labels 不是固定字段。

不同 bank 可能定义：

- `topic`
- `importance`
- `customer_stage`
- `product`
- `mood`

也可能完全不同。

如果写死在代码里：

- schema 就没法跟随配置变化

所以这里必须运行时根据配置生成：

- prompt 文本
- Pydantic 模型
- JSON schema

这样 LLM 的 `response_format` 才能和当前 bank 的 labels 配置一致。

## `free_form_entities` 的作用

这也是一个容易误会的点。

如果：

```python
free_form_entities = False
```

系统会走 labels-only 思路：

- 模型仍然可以输出 `labels`
- 但不希望它输出普通自由实体

所以 schema 里会把 `entities` 保留成：

```python
list[Entity] | None
```

但描述写成：

```text
Leave empty — labels-only mode
```

这个设计不是为了删除字段，而是为了：

- 保持返回结构兼容
- 同时明确约束模型不要填普通实体

## 为什么还要处理 `required`

这几行很关键：

```python
base_extra = base_fact_class.model_config.get("json_schema_extra")
base_required = ...
```

原因是：

- 基础 fact class 已经通过 `json_schema_extra` 定义了一组 required 字段
- 现在又动态新增了 `labels`
- 如果不把旧的 required 继承过来，动态模型生成后的 JSON schema 里可能只剩新字段，或者 required 信息丢失

所以这里要显式做：

1. 先继承父类 required
2. 再把 `labels` 追加进去

这样最终 schema 才完整。

## 可以把这个函数理解成一个“组合器”

从工程上看，这个函数最适合理解成：

```text
最终 prompt
= 基础 prompt(mode)
+ retain_mission section（可选）
+ causal section（可选）
+ labels section（可选）

最终 response schema
= 基础 schema(mode + causal)
+ dynamic labels schema（可选）
+ entities-only/labels-only 约束（可选）
```

也就是说，它不是单纯“if/else 选 prompt”，而是在做一套组合。

## 为什么它看起来复杂，但其实边界很清晰

复杂感主要来自“一个函数里同时处理了 prompt 和 schema”。

但它的边界其实很明确：

- 它不负责决定是否走 LLM
- 它只负责在“已经确定要走 LLM”时，给出一套匹配的输入约束和输出约束

所以你可以这样记：

- `chunks` 模式：外层短路，不进这里
- 非 `chunks` 模式：进这里装配 prompt + schema
- `verbatim`：特殊的 metadata-only 路线
- `labels`：在基础 schema 上再做动态扩展

## 如果你问“最常见分支到底是哪条”

从代码结构上看，最普通的一条 LLM 路径是：

1. `retain_extraction_mode != "chunks"`
2. 也不是 `custom` / `verbose` / `verbatim`
3. `retain_extract_causal_links` 看配置决定开或关
4. 没有 `entity_labels`

这时 `_build_extraction_prompt_and_schema()` 基本就是：

- 用 `CONCISE_FACT_EXTRACTION_PROMPT`
- 可选追加 causal section
- 不追加 labels section
- 返回基础 response schema

而如果你说“生产里最常见的 retain 模式是不是 `chunks`”，那是另一个层面的问题：

- 如果答案是是的，那么这里这个函数压根不会参与那批请求
- 所以不能把“系统最常用模式”与“这个函数内部最常走哪条分支”混为一谈

## 你看这个函数时，建议用这张心智图

```text
先问：这次是不是 chunks？
  是 -> 外层直接返回，不看这个函数
  否 -> 进入这个函数

进入后再问：
  1. retain_mission 要不要加？
  2. extraction_mode 是哪种？
  3. causal links 要不要加？
  4. labels 要不要加？
  5. entities 是 free-form 还是 labels-only？
```

这样读就不会乱。
