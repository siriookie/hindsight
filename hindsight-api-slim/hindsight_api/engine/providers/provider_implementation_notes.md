# LLM Provider 实现细节说明

文件范围：

- [anthropic_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py)
- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py)
- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py)
- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py)
- [litellm_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/litellm_llm.py)
- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py)
- [llm_wrapper.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/llm_wrapper.py)

这份文档解释这些 provider 的实现细节，以及它们在调用方式上的共同点和差异。

## 总体架构

这一层的基本结构是：

1. 上层代码通常持有 `LLMProvider` / `ConfiguredLLMProvider`
2. 通过 [llm_wrapper.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/llm_wrapper.py) 的 `create_llm_provider(...)` 根据 `provider` 字符串创建具体实现
3. 具体实现类都继承 `LLMInterface`
4. 每个 provider 自己实现：
   - `verify_connection()`
   - `call()`
   - `call_with_tools()`

也就是说，公共抽象是统一的，但每个 provider 内部调用的 SDK、鉴权方式、消息格式转换、工具调用细节都不一样。

## provider 分发在哪里

provider 到实现类的分发在：

- [llm_wrapper.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/llm_wrapper.py#L141)

大致映射关系是：

- `anthropic` -> `AnthropicLLM`
- `claude-code` -> `ClaudeCodeLLM`
- `openai-codex` -> `CodexLLM`
- `gemini` / `vertexai` -> `GeminiLLM`
- `litellm` / `bedrock` -> `LiteLLMLLM`
- `openai` / `groq` / `ollama` / `lmstudio` / `minimax` / `volcano` -> `OpenAICompatibleLLM`

所以：

- `bedrock` 在这里不是单独类，而是走 `LiteLLMLLM`
- `vertexai` 也不是单独类，而是走 `GeminiLLM`

## 共同点

虽然实现不同，但这几个 provider 有一组很明显的共性。

### 1. 都实现同样的抽象接口

这些类都继承 `LLMInterface`，对上层暴露相同风格的方法：

- `verify_connection()`
- `call()`
- `call_with_tools()`

这意味着上层 retain / recall / reflect 基本不关心底层具体是哪家 provider。

### 2. 都支持“普通文本调用”和“结构化输出”

所有这些实现的 `call()` 基本都支持：

- `messages`
- `response_format`
- `max_completion_tokens`
- `temperature`
- `scope`
- `max_retries`
- `initial_backoff`
- `max_backoff`
- `skip_validation`
- `return_usage`

即使底层 SDK 不完全支持这些能力，也会尽量做兼容。

### 3. 都做重试

这些 provider 的 `call()` 基本都有：

- retry loop
- 指数退避
- auth 错误快速失败
- JSON parse 失败重试
- 连接 / 限流 / 5xx 错误重试

只是重试条件和异常类型判断各不相同。

### 4. 都做 metrics / tracing

成功调用后，通常都会记录：

- metrics
- tracing span
- 慢调用日志

只是 token usage 的来源有些是精确值，有些是估算值。

### 5. 都会把 structured output 最终收敛到同样的用法

大多数 provider 都遵循这种套路：

1. 如果传了 `response_format`
2. 想办法让模型输出 JSON
3. 解析 JSON
4. 如果 `skip_validation=True`，直接返回 dict
5. 否则 `response_format.model_validate(...)`

差异只在于“如何逼模型产出 JSON”。

---

## AnthropicLLM

文件：

- [anthropic_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py)

### 核心定位

这是标准 Anthropic API 的实现，直接使用 Anthropic 官方 SDK。

### 鉴权方式

- 用 API key
- 初始化时创建 `AsyncAnthropic`

关键位置：

- [anthropic_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py#L61)

### 普通调用方式

`call()` 的核心是：

- 把 OpenAI 风格的 `messages` 转成 Anthropic 风格
- 将 `system` 消息单独提取为 `system_prompt`
- 用 `self._client.messages.create(...)` 发请求

关键位置：

- [anthropic_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py#L95)
- [anthropic_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py#L179)

### 结构化输出方式

Anthropic 这里不是原生 strict schema 路线，而是：

- 如果有 `response_format`
- 把 schema 文字拼到 system prompt 中
- 再自己解析返回内容中的 JSON

这意味着：

- schema enforcement 更偏“软约束”
- 不是像 OpenAI strict json schema 那样由 API 级强约束

### usage 统计

Anthropic 可以拿到真实 usage：

- `input_tokens`
- `output_tokens`

这是精确值，不是估算。

### tool calling

`call_with_tools()` 也是直接走 Anthropic API。

关键位置：

- [anthropic_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/anthropic_llm.py#L302)

### 特点总结

- 优点：标准、稳定、usage 精确、SDK 原生
- 缺点：structured output 仍依赖 prompt + 手工 JSON 解析，不是严格 schema API 风格

---

## ClaudeCodeLLM

文件：

- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py)

### 核心定位

这是走 Claude Code / Claude Agent SDK 的实现，不是直接打 Anthropic API。

### 鉴权方式

- 不使用 API key
- 依赖本机 `claude auth login`
- 通过 `claude_agent_sdk` 使用当前机器上的 Claude 登录态

关键位置：

- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L39)
- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L71)

### 普通调用方式

`call()` 的核心是：

- 用 `claude_agent_sdk.query(...)`
- 把 system prompt 和 user 内容分开整理
- assistant 历史消息并不是真正多轮对话结构，而是被拼进 user 文本里

关键位置：

- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L106)
- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L166)
- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L193)

### 结构化输出方式

和 Anthropic 类似，也是：

- 把 JSON schema 说明写进 prompt
- 要求只输出 JSON
- 再手工 `json.loads(...)`

### usage 统计

Claude Agent SDK 不返回精确 token 统计。

所以这里是估算：

- 输入字符数 / 4
- 输出字符数 / 4

关键位置：

- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L231)

### tool calling

这是它和 `AnthropicLLM` 差异最大的地方之一。

普通 `query()` 不够用，所以 `call_with_tools()` 改走 `ClaudeSDKClient` 路线。

关键位置：

- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L306)
- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L321)
- [claude_code_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/claude_code_llm.py#L479)

另外，`tool_choice` 也不是底层原生支持，所以这里自己模拟：

- `required`
- `none`
- 特定函数名

### 特点总结

- 优点：可直接利用 Claude Code 登录态，不必单独买 API credits
- 缺点：多轮消息支持较弱，usage 是估算，tools 支持靠 SDK 侧绕路

---

## CodexLLM

文件：

- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py)

### 核心定位

这是走 Codex CLI / ChatGPT 账号 OAuth 的实现，不是标准 OpenAI Platform API。

### 鉴权方式

- 从 `~/.codex/auth.json` 读取 OAuth 凭证
- 依赖 `codex auth login`

关键位置：

- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L42)
- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L69)

### 普通调用方式

它不是 SDK 调用，而是：

- 直接用 `httpx.AsyncClient`
- 向 ChatGPT backend 的 `/codex/responses` 发请求
- 使用 SSE 流式返回

关键位置：

- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L174)
- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L244)
- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L329)

### structured output 方式

它也是：

- 把 schema 文字注入 instructions
- 收到 SSE 文本后自己拼接
- 手工 JSON parse

### reasoning 处理

Codex 有一套自己的 reasoning summary 概念：

- `concise`
- `auto`
- `detailed`

这里会把通用 reasoning effort 映射成 Codex 自己的格式。

关键位置：

- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L98)

### usage 统计

Codex SSE 不提供精确 token usage。

这里的 usage 也是估算值：

- 输入字符数 / 4
- 输出字符数 / 4

### tool calling

`call_with_tools()` 也是走 `/codex/responses` SSE。

特点是：

- 先把 OpenAI 格式 tools 转成 Codex 需要的扁平格式
- 还要把 `tool_choice` 做一次 normalize

关键位置：

- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L129)
- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L431)
- [codex_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/codex_llm.py#L515)

### 特点总结

- 优点：可以用 ChatGPT Plus/Pro 登录态，不依赖 OpenAI Platform API key
- 缺点：不是标准 API，usage 只能估算，SSE 解析逻辑更重

---

## GeminiLLM

文件：

- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py)

### 核心定位

这是 Google Gemini 和 Vertex AI 的统一实现。

同一个类支持两种模式：

- `provider="gemini"`：Gemini API
- `provider="vertexai"`：Vertex AI

### 鉴权方式

Gemini API：

- API key

Vertex AI：

- ADC
- 或 service account
- 或预先传入 credentials 对象

关键位置：

- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L42)
- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L68)

### 普通调用方式

使用 Google `genai.Client`：

- `self._client.aio.models.generate_content(...)`

并且会把 OpenAI-style messages 转成 Gemini 的 `Content/Part` 结构。

关键位置：

- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L247)

### structured output 方式

Gemini 是这几个实现里对 structured output 支持最“像原生 schema”的之一。

它会设置：

- `response_mime_type = "application/json"`
- `response_schema = response_format`

关键位置：

- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L225)
- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L226)

然后再用 `parse_llm_json(...)` 做兼容解析。

### safety settings

Gemini/VertexAI 还有一个特殊点：

- 支持 per-request safety settings
- 通过 `ContextVar` 从 `ConfiguredLLMProvider` 注入

这在其他 provider 里没有对应能力。

### usage 统计

Gemini 可以从 `usage_metadata` 里取到 token 数。

所以这里是较精确的 usage，不是字符估算。

### tool calling

Gemini 的 `call_with_tools()` 会做一大段消息/工具结构转换：

- OpenAI tools -> Gemini `FunctionDeclaration`
- OpenAI messages -> Gemini `Content/Part`
- OpenAI tool_choice -> Gemini `FunctionCallingConfig`

关键位置：

- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L383)
- [gemini_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/gemini_llm.py#L494)

### 特点总结

- 优点：同时支持 Gemini API 和 Vertex AI；structured output 支持较自然；usage 较准确
- 缺点：消息和 tools 格式转换最复杂，Google SDK 类型系统也更重

---

## LiteLLMLLM

文件：

- [litellm_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/litellm_llm.py)

### 核心定位

这是“通用代理层” provider。

不是某一家模型的原生实现，而是通过 LiteLLM 去打很多兼容后端。

典型场景：

- Bedrock
- Azure OpenAI
- Together AI
- Fireworks
- 以及其它 LiteLLM 支持的 provider

### 鉴权方式

由 LiteLLM 负责，可能是：

- API key
- AWS/Boto credential chain
- 其他 provider-specific 机制

### 普通调用方式

核心调用是：

- `litellm.acompletion(...)`

关键位置：

- [litellm_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/litellm_llm.py#L145)

### 参数构造方式

它先统一走 `_build_common_kwargs(...)`：

- model
- messages
- timeout
- api_key
- api_base
- max_completion_tokens
- temperature

### structured output 方式

如果有 `response_format`，它会尝试走 LiteLLM 的：

- `response_format = {"type": "json_schema", ...}`

所以这里更接近 OpenAI 风格，而不是纯 prompt 注入。

### usage 统计

通常可以从 LiteLLM 返回中拿到 usage：

- `prompt_tokens`
- `completion_tokens`

精度取决于 LiteLLM 下游 provider 的透传质量。

### tool calling

`call_with_tools()` 也是直接走：

- `litellm.acompletion(...)`

并透传：

- `tools`
- `tool_choice`

关键位置：

- [litellm_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/litellm_llm.py#L266)

### 特点总结

- 优点：覆盖面最广，适合快速接入大量 provider
- 缺点：能力与行为高度依赖 LiteLLM 和下游 provider 组合，语义一致性最弱

---

## OpenAICompatibleLLM

文件：

- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py)

### 核心定位

这是“OpenAI 协议兼容”的统一实现。

覆盖：

- OpenAI
- Groq
- Ollama
- LM Studio
- MiniMax
- Volcano

### 鉴权方式

取决于 provider：

- OpenAI / Groq / MiniMax：API key
- Ollama / LM Studio：本地服务，可用 dummy key

### 普通调用方式

核心是：

- `AsyncOpenAI`
- `self._client.chat.completions.create(...)`

关键位置：

- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py#L128)
- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py#L242)

### 特殊分支

它是这几个实现里分支最多的一个，因为不同兼容 provider 行为差别很大。

#### 1. reasoning models

它会识别：

- GPT-5
- o1
- o3
- deepseek

并为这些模型设置：

- `reasoning_effort`
- 最小 token 上限
- reasoning-specific token cap

#### 2. provider-specific extra_body

例如：

- Groq 的 seed
- Groq 的 service_tier
- reasoning 相关附加参数

#### 3. Ollama native structured output

如果：

- provider == `ollama`
- 且请求 structured output

就不走 OpenAI chat completions，而改走 `_call_ollama_native(...)`。

关键位置：

- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py#L234)
- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py#L687)

### structured output 方式

它有两种：

#### strict schema

如果：

- `strict_schema=True`

则用 OpenAI 风格严格 JSON schema：

- `response_format = {"type": "json_schema", ...}`

#### soft schema

否则：

- 把 schema 文本注入 prompt
- 再尽量使用 `json_object`

这比 Anthropic / Claude Code 更强，但又兼容一些不支持 strict schema 的兼容后端。

### usage 统计

一般能拿到精确 usage：

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

这是它的一大优势。

### tool calling

`call_with_tools()` 也是标准 OpenAI-compatible 路线。

同时会做一层兼容转换：

- 如果 `tool_choice` 是指定函数 dict
- 某些 provider 不支持这种格式
- 就转换成 `required + 过滤 tools`

关键位置：

- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py#L527)
- [openai_compatible_llm.py](/E:/py/hindsight/hindsight-api-slim/hindsight_api/engine/providers/openai_compatible_llm.py#L558)

### 特点总结

- 优点：最通用、最成熟、usage 精确、对 reasoning models 支持最好
- 缺点：为了兼容很多 provider，内部条件分支较多

---

## 它们的共同调用流程

尽管底层不同，大体都可以抽象成同一个五步流程：

1. 接受统一的 `messages` 和控制参数
2. 按 provider 需要把消息格式转换成底层 SDK / API 能接受的结构
3. 如果要 structured output，就附加 schema 约束
4. 调用底层 API / SDK
5. 把结果统一转回：
   - 纯文本
   - 结构化对象
   - `LLMToolCallResult`

所以它们共同点是“输入输出协议统一”，不同点是“中间适配层实现不同”。

## 它们的主要不同点

### 1. 鉴权方式不同

- `AnthropicLLM`：Anthropic API key
- `ClaudeCodeLLM`：`claude auth login`
- `CodexLLM`：`~/.codex/auth.json` OAuth
- `GeminiLLM`：Gemini API key 或 Vertex AI credentials
- `LiteLLMLLM`：由 LiteLLM / 下游 provider 决定
- `OpenAICompatibleLLM`：API key 或本地 dummy key

### 2. 底层调用方式不同

- `AnthropicLLM`：Anthropic SDK `messages.create`
- `ClaudeCodeLLM`：Claude Agent SDK `query()` / `ClaudeSDKClient`
- `CodexLLM`：HTTP + SSE `/codex/responses`
- `GeminiLLM`：Google `generate_content`
- `LiteLLMLLM`：`litellm.acompletion`
- `OpenAICompatibleLLM`：`chat.completions.create`

### 3. structured output 强度不同

最偏原生 schema：

- `GeminiLLM`
- `OpenAICompatibleLLM`（strict schema 可用时）
- `LiteLLMLLM`（前提是下游支持）

偏 prompt 注入 + 手工 JSON parse：

- `AnthropicLLM`
- `ClaudeCodeLLM`
- `CodexLLM`

### 4. usage 精度不同

精确 usage：

- `AnthropicLLM`
- `GeminiLLM`
- `OpenAICompatibleLLM`
- `LiteLLMLLM`（视下游而定，通常可拿到）

估算 usage：

- `ClaudeCodeLLM`
- `CodexLLM`

### 5. tool calling 成熟度不同

更原生：

- `AnthropicLLM`
- `GeminiLLM`
- `OpenAICompatibleLLM`
- `LiteLLMLLM`

需要额外绕路：

- `ClaudeCodeLLM`
- `CodexLLM`

### 6. 消息格式转换复杂度不同

最简单：

- `LiteLLMLLM`
- `OpenAICompatibleLLM`

中等：

- `AnthropicLLM`

最复杂：

- `GeminiLLM`
- `ClaudeCodeLLM`
- `CodexLLM`

---

## 可以怎么理解这些实现的设计取向

如果按设计目标来分：

### 标准官方 API provider

- `AnthropicLLM`
- `GeminiLLM`

特点：

- 用官方 SDK
- 行为稳定
- usage 更靠谱

### 账号登录态 provider

- `ClaudeCodeLLM`
- `CodexLLM`

特点：

- 适合利用本机 CLI 登录态
- 不一定依赖 API credits
- 但调用语义、usage、tool 支持通常更绕

### 兼容层 provider

- `LiteLLMLLM`
- `OpenAICompatibleLLM`

特点：

- 覆盖面广
- 兼容性强
- 但内部有更多 provider-specific 条件分支

---

## 如果只看“最稳定的服务端使用”

通常优先级大致会是：

- 需要 Anthropic 官方能力：`AnthropicLLM`
- 需要 Gemini / Vertex：`GeminiLLM`
- 需要 OpenAI/Groq/Ollama/LM Studio/MiniMax：`OpenAICompatibleLLM`
- 需要一层通用代理：`LiteLLMLLM`
- 需要复用本机订阅登录态：`ClaudeCodeLLM` / `CodexLLM`

## 如果只看“实现复杂度”

大致从简单到复杂可以粗略理解为：

1. `LiteLLMLLM`
2. `AnthropicLLM`
3. `OpenAICompatibleLLM`
4. `GeminiLLM`
5. `CodexLLM`
6. `ClaudeCodeLLM`

这不是能力排序，而是“内部适配工作量”的粗略感觉。

## 最后一句总结

这些 provider 的核心思想不是“每家都重新定义一套接口”，而是：

- 外层接口统一
- 内层尽可能贴近各自真实调用方式
- 在 wrapper 层把差异吸收掉

所以你读代码时，最有用的心智模型是：

- 先看这个 provider 的“鉴权来源”
- 再看它的“底层 SDK / API”
- 再看它是怎么做 structured output 和 tool calling 的

这样差异就会非常清楚。
