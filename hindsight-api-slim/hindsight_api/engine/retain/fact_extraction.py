"""
Fact extraction from text using LLM.

Extracts semantic facts, entities, and temporal information from text.
Uses the LLMConfig wrapper for all LLM calls.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from ...config import get_config
from ..llm_wrapper import LLMConfig, OutputTooLongError, sanitize_llm_output
from ..response_models import TokenUsage
from .entity_labels import (
    EntityLabelsConfig,
    build_labels_lookup,
    build_labels_model,
    is_label_entity,
    parse_entity_labels,
)


def _infer_temporal_date(fact_text: str, event_date: datetime | None) -> str | None:
    """
    Infer a temporal date from fact text when LLM didn't provide occurred_start.

    This is a fallback for when the LLM fails to extract temporal information
    from relative time expressions like "last night", "yesterday", etc.
    """
    if event_date is None:
        return None

    fact_lower = fact_text.lower()

    # Map relative time expressions to day offsets
    temporal_patterns = {
        r"\blast night\b": -1,
        r"\byesterday\b": -1,
        r"\btoday\b": 0,
        r"\bthis morning\b": 0,
        r"\bthis afternoon\b": 0,
        r"\bthis evening\b": 0,
        r"\btonigh?t\b": 0,
        r"\btomorrow\b": 1,
        r"\blast week\b": -7,
        r"\bthis week\b": 0,
        r"\bnext week\b": 7,
        r"\blast month\b": -30,
        r"\bthis month\b": 0,
        r"\bnext month\b": 30,
    }

    for pattern, offset_days in temporal_patterns.items():
        if re.search(pattern, fact_lower):
            target_date = event_date + timedelta(days=offset_days)
            return target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # If no relative time expression found, return None
    return None


def _sanitize_text(text: str | None) -> str | None:
    return sanitize_llm_output(text)


class Entity(BaseModel):
    """An entity extracted from text."""

    text: str = Field(
        description="The specific, named entity as it appears in the fact. Must be a proper noun or specific identifier."
    )


class Fact(BaseModel):
    """
    Final fact model for storage - built from lenient parsing of LLM response.

    This is what fact_extraction returns and what the rest of the pipeline expects.
    Combined fact text format: "what | when | where | who | why"
    """

    # Required fields
    fact: str = Field(description="Combined fact text: what | when | where | who | why")
    fact_type: Literal["world", "experience", "opinion"] = Field(description="Perspective: world/experience/opinion")

    # Optional temporal fields
    occurred_start: str | None = None
    occurred_end: str | None = None

    # Optional location field
    where: str | None = Field(
        None, description="WHERE the fact occurred or is about (specific location, place, or area)"
    )

    # Optional structured data
    entities: list[Entity] | None = None
    causal_relations: list["CausalRelation"] | None = None


class CausalRelation(BaseModel):
    """Causal relationship from this fact to a previous fact (stored format)."""

    target_fact_index: int = Field(description="Index of the related fact in the facts array (0-based).")
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target: 'caused_by' = this fact was caused by the target"
    )
    strength: float = Field(
        description="Strength of relationship (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class FactCausalRelation(BaseModel):
    """
    Causal relationship from this fact to a PREVIOUS fact (embedded in each fact).

    Uses index-based references but ONLY allows referencing facts that appear
    BEFORE this fact in the list. This prevents hallucination of invalid indices.
    """

    target_index: int = Field(
        description="Index of the PREVIOUS fact this relates to (0-based). "
        "MUST be less than this fact's position in the list. "
        "Example: if this is fact #5, target_index can only be 0, 1, 2, 3, or 4."
    )
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target fact: 'caused_by' = this fact was caused by the target fact"
    )
    strength: float = Field(
        description="Strength of relationship (0.0 to 1.0). 1.0 = strong, 0.5 = moderate",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class ExtractedFact(BaseModel):
    """A single extracted fact."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )
    #  what: str   # 核心事实（1-2句话）
    #     when: str   # 时间（ISO 日期 or "N/A"）
    #     where: str  # 地点
    #     who: str    # 涉及人员（解析共指：如"我室友 Emily"→"Emily，用户的室友"）
    #     why: str    # 背景/动机/情感
    what: str = Field(description="Core fact - concise but complete (1-2 sentences)")
    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")
    why: str = Field(description="Context/significance if important. 'N/A' if obvious.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    # ISO 时间戳
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    #  world=客观, assistant=第一人称
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts. 'assistant' = first-person actions, experiences, or observations by the speaker."
    )
    entities: list[Entity] | None = Field(default=None, description="People, places, concepts")
    # 与之前 fact 的因果关系
    causal_relations: list[FactCausalRelation] | None = Field(
        default=None, description="Links to previous facts (target_index < this fact's index)"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        """Ensure entities is always a list (convert None to empty list)."""
        if v is None:
            return []
        return v

    def build_fact_text(self) -> str:
        """Combine all dimensions into a single comprehensive fact string."""
        parts = [self.what]

        # Add 'who' if not N/A
        if self.who and self.who.upper() != "N/A":
            parts.append(f"Involving: {self.who}")

        # Add 'why' if not N/A
        if self.why and self.why.upper() != "N/A":
            parts.append(self.why)

        if len(parts) == 1:
            return parts[0]

        return " | ".join(parts)


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts (causal relations are embedded in each fact)."""

    facts: list[ExtractedFact] = Field(description="List of extracted factual statements")


class ExtractedFactVerbose(BaseModel):
    """A single extracted fact with verbose field descriptions for detailed extraction."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    what: str = Field(
        description="WHAT happened - COMPLETE, DETAILED description with ALL specifics. "
        "NEVER summarize or omit details. Include: exact actions, objects, quantities, specifics. "
        "BE VERBOSE - capture every detail that was mentioned. "
        "Example: 'Emily got married to Sarah at a rooftop garden ceremony with 50 guests attending and a live jazz band playing' "
        "NOT: 'A wedding happened' or 'Emily got married'"
    )

    when: str = Field(
        description="WHEN it happened - ALWAYS include temporal information if mentioned. "
        "Include: specific dates, times, durations, relative time references. "
        "Examples: 'on June 15th, 2024 at 3pm', 'last weekend', 'for the past 3 years', 'every morning at 6am'. "
        "Write 'N/A' ONLY if absolutely no temporal context exists. Prefer converting to absolute dates when possible."
    )

    where: str = Field(
        description="WHERE it happened or is about - SPECIFIC locations, places, areas, regions if applicable. "
        "Include: cities, neighborhoods, venues, buildings, countries, specific addresses when mentioned. "
        "Examples: 'downtown San Francisco at a rooftop garden venue', 'at the user's home in Brooklyn', 'online via Zoom', 'Paris, France'. "
        "Write 'N/A' ONLY if absolutely no location context exists or if the fact is completely location-agnostic."
    )

    who: str = Field(
        description="WHO is involved - ALL people/entities with FULL context and relationships. "
        "Include: names, roles, relationships to user, background details. "
        "Resolve coreferences (if 'my roommate' is later named 'Emily', write 'Emily, the user's college roommate'). "
        "BE DETAILED about relationships and roles. "
        "Example: 'Emily (user's college roommate from Stanford, now works at Google), Sarah (Emily's partner of 5 years, software engineer)' "
        "NOT: 'my friend' or 'Emily and Sarah'"
    )

    why: str = Field(
        description="WHY it matters - ALL emotional, contextual, and motivational details. "
        "Include EVERYTHING: feelings, preferences, motivations, observations, context, background, significance. "
        "BE VERBOSE - capture all the nuance and meaning. "
        "FOR ASSISTANT FACTS: MUST include what the user asked/requested that led to this interaction! "
        "Example (world): 'The user felt thrilled and inspired, has always dreamed of an outdoor ceremony, mentioned wanting a similar garden venue, was particularly moved by the intimate atmosphere and personal vows' "
        "Example (assistant): 'User asked how to fix slow API performance with 1000+ concurrent users, expected 70-80% reduction in database load' "
        "NOT: 'User liked it' or 'To help user'"
    )

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence (set occurred dates), 'conversation' = general info (no occurred dates)",
    )

    occurred_start: str | None = Field(
        default=None,
        description="WHEN the event happened (ISO timestamp). Only for fact_kind='event'. Leave null for conversations.",
    )
    occurred_end: str | None = Field(
        default=None,
        description="WHEN the event ended (ISO timestamp). Only for events with duration. Leave null for conversations.",
    )

    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts about other people, events, general knowledge. 'assistant' = first-person actions, experiences, or observations by the speaker (e.g., 'I changed X', 'I discovered Y')."
    )

    entities: list[Entity] | None = Field(
        default=None,
        description="Named entities, objects, AND abstract concepts from the fact. Include: people names, organizations, places, significant objects (e.g., 'coffee maker', 'car'), AND abstract concepts/themes (e.g., 'friendship', 'career growth', 'loss', 'celebration'). Extract anything that could help link related facts together.",
    )

    causal_relations: list[FactCausalRelation] | None = Field(
        default=None,
        description="Causal links to PREVIOUS facts only. target_index MUST be less than this fact's position. "
        "Example: fact #3 can only reference facts 0, 1, or 2. Max 2 relations per fact.",
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        if v is None:
            return []
        return v


class FactExtractionResponseVerbose(BaseModel):
    """Response for verbose fact extraction."""

    facts: list[ExtractedFactVerbose] = Field(description="List of extracted factual statements")


class ExtractedFactNoCausal(BaseModel):
    """A single extracted fact WITHOUT causal relations (for when causal extraction is disabled)."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    # Same fields as ExtractedFact but without causal_relations
    what: str = Field(description="WHAT happened - COMPLETE, DETAILED description with ALL specifics.")
    when: str = Field(description="WHEN it happened - include temporal information if mentioned.")
    where: str = Field(description="WHERE it happened - SPECIFIC locations if applicable.")
    who: str = Field(description="WHO is involved - ALL people/entities with relationships.")
    why: str = Field(description="WHY it matters - emotional, contextual, and motivational details.")

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence, 'conversation' = general info",
    )
    occurred_start: str | None = Field(default=None, description="WHEN the event happened (ISO timestamp).")
    occurred_end: str | None = Field(default=None, description="WHEN the event ended (ISO timestamp).")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = about the user/others. 'assistant' = experience with assistant."
    )
    entities: list[Entity] | None = Field(
        default=None,
        description="Named entities, objects, and concepts from the fact.",
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        if v is None:
            return []
        return v


class FactExtractionResponseNoCausal(BaseModel):
    """Response for fact extraction without causal relations."""

    facts: list[ExtractedFactNoCausal] = Field(description="List of extracted factual statements")


class VerbatimExtractedFact(BaseModel):
    """
    Schema for verbatim extraction mode.

    Omits 'what' entirely — the original chunk text is used as fact_text in code.
    The LLM only extracts metadata: entities, temporal info, location, people.
    """

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["when", "where", "who", "fact_type"]},
    )

    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts. 'assistant' = first-person actions, experiences, or observations by the speaker."
    )
    entities: list[Entity] | None = Field(default=None, description="People, places, concepts")

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        if v is None:
            return []
        return v


class VerbatimFactExtractionResponse(BaseModel):
    """Response for verbatim extraction mode (one entry per chunk, no fact text)."""

    facts: list[VerbatimExtractedFact] = Field(description="List of metadata entries (one per chunk)")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks, preserving conversation structure when possible.

    For JSON conversation arrays (user/assistant turns), splits at turn boundaries
    while preserving speaker context. For plain text, uses sentence-aware splitting.

    Args:
        text: Input text to chunk (plain text or JSON conversation)
        max_chars: Maximum characters per chunk (default 120k ≈ 30k tokens)

    Returns:
        List of text chunks, roughly under max_chars

    实现原理：这是 retain 阶段最底层的切块函数，目标不是做“最聪明的语义切分”，
    而是做“稳定、可恢复、足够保守”的切分：
    1. 小文本不切，直接整块返回
    2. 如果文本长得像 JSON 对话数组，就按 turn 边界切
    3. 否则按段落/换行/句子/词语逐级退化切分

    这样既能尽量保留自然边界，又能保证 chunk 切分规则足够稳定，方便 delta retain
    用 `chunk_index + content_hash` 识别哪些块发生了变化。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 小文本直接保留为一个 chunk，避免不必要的切分导致索引漂移。
    if len(text) <= max_chars:
        return [text]

    # 会话 JSON 需要优先走专门路径，因为普通句子切分会破坏对话 turn 边界，
    # 影响后续 fact 提取中的说话者语义。
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
            # This looks like a conversation - chunk at turn boundaries
            return _chunk_conversation(parsed, max_chars)
    except (json.JSONDecodeError, ValueError):
        pass

    # 普通文本走递归字符切分，但优先按“更自然的边界”切，而不是直接硬按字符数截断。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "! ",  # Exclamations
            "? ",  # Questions
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Words
            "",  # Characters (last resort)
        ],
    )

    return splitter.split_text(text)


def _chunk_conversation(turns: list[dict], max_chars: int) -> list[str]:
    """
    Chunk a conversation array at turn boundaries, preserving complete turns.

    Args:
        turns: List of conversation turn dicts (with 'role' and 'content' keys)
        max_chars: Maximum characters per chunk

    Returns:
        List of JSON-serialized chunks, each containing complete turns
    """

    chunks = []
    current_chunk = []
    current_size = 2  # Account for "[]"

    for turn in turns:
        # Estimate size of this turn when serialized (with comma separator)
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_size = len(turn_json) + 1  # +1 for comma

        # If adding this turn would exceed limit and we have turns, save current chunk
        if current_size + turn_size > max_chars and current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = []
            current_size = 2  # Reset to "[]"

        # Add turn to current chunk
        current_chunk.append(turn)
        current_size += turn_size

    # Add final chunk if non-empty
    if current_chunk:
        chunks.append(json.dumps(current_chunk, ensure_ascii=False))

    return chunks if chunks else [json.dumps(turns, ensure_ascii=False)]


# =============================================================================
# FACT EXTRACTION PROMPTS
# =============================================================================

# Base prompt template (shared by concise and custom modes)
# Uses {extraction_guidelines} placeholder for mode-specific instructions
# 中文翻译：
# 从文本中提取“重要事实”。要有选择性——只提取值得长期记住的事实。
#
# 语言要求（强制）：
# 检测输入文本的语言，并且你的所有输出必须使用完全相同的语言。
# 严禁翻译或切换到任何其他语言。在任何情况下都不得使用不同语言输出。
# 你的每一个词都必须与输入语言一致。
#
# {retain_mission_section}{extraction_guidelines}
#
# ══════════════════════════════════════════════════════════════════════════
# 事实格式 —— 保持简洁
# ══════════════════════════════════════════════════════════════════════════
#
# 1. **what**：核心事实——简洁但完整（最多1-2句话）
# 2. **when**：如果提到时间信息则填写；否则写 "N/A"。如果知道具体星期几要写出
# 3. **where**：如果相关则填写地点；否则写 "N/A"
# 4. **who**：涉及的人及其关系；如果只是一般信息则写 "N/A"
# 5. **why**：仅在重要时填写背景/意义；如果显而易见则写 "N/A"
#
# 简洁性：
# 抓住本质，而不是逐字复述。一句高质量的话胜过三句普通的话。
#
# ══════════════════════════════════════════════════════════════════════════
# 指代消解（共指消解）
# ══════════════════════════════════════════════════════════════════════════
#
# 当泛指和具体名字同时出现时要进行关联：
#
# - "my roommate" + "Emily" → 使用 "Emily（用户的室友）"
# - "the manager" + "Sarah" → 使用 "Sarah（经理）"
#
# ══════════════════════════════════════════════════════════════════════════
# 分类
# ══════════════════════════════════════════════════════════════════════════
#
# fact_kind（事实类型）：
#
# - "event"：可标定时间的具体事件（需要设置 occurred_start / occurred_end）
# - "conversation"：持续性的状态、偏好、特征（不设置日期）
#
# fact_type（事实归属）：
#
# - "world"：关于他人、外部事件、通用知识、客观事实
# - "assistant"：第一人称的行为、经历或观察（例如：“我修改了X”，“我发现了Y”，“我调试了Z”）
#   也包括与用户的交互（请求、推荐）。
#   只要叙述者描述的是自己做过、尝试过、学到或决定的事情——使用 "assistant"
#
# ══════════════════════════════════════════════════════════════════════════
# 时间处理
# ══════════════════════════════════════════════════════════════════════════
#
# 使用输入中的“事件日期（Event Date）”作为参考来解析相对时间。
#
# - 关键要求：必须将所有相对时间表达转换为绝对日期，并写入事实文本中
#   "yesterday" → 写成具体日期（例如："on November 12, 2024"），不能写“yesterday”
#   "last night"、"this morning"、"today"、"tonight" → 全部转换为具体日期
#
# - 对于事件（event）：
#   必须设置 occurred_start 和 occurred_end（点事件两者相同）
#
# - 对于对话类事实（conversation）：
#   不设置 occurred 日期
#
# ══════════════════════════════════════════════════════════════════════════
# 实体
# ══════════════════════════════════════════════════════════════════════════
#
# 应包含：
#
# - 人名
# - 组织
# - 地点
# - 关键对象
# - 抽象概念（如职业、友谊等）
#
# 如果事实涉及用户，必须始终包含 "user"。
#
# {examples}
_BASE_FACT_EXTRACTION_PROMPT = """Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

LANGUAGE: MANDATORY — Detect the language of the input text and produce ALL output in that EXACT same language. You are STRICTLY FORBIDDEN from translating or switching to any other language. Every single word of your output must be in the same language as the input. Do NOT output in a different language under any circumstance.

{retain_mission_section}{extraction_guidelines}

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. **what**: Core fact - concise but complete (1-2 sentences max)
2. **when**: Temporal info if mentioned. "N/A" if none. Use day name when known.
3. **where**: Location if relevant. "N/A" if none.
4. **who**: People involved with relationships. "N/A" if just general info.
5. **why**: Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": About other people, external events, general knowledge, objective facts
- "assistant": First-person actions, experiences, or observations by the speaker/author (e.g., "I changed X", "I discovered Y", "I debugged Z"). Also includes interactions with the user (requests, recommendations). If the narrator describes something they did, tried, learned, or decided — use "assistant".

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- CRITICAL: Convert ALL relative temporal expressions to absolute dates in the fact text itself.
  "yesterday" → write the resolved date (e.g. "on November 12, 2024"), NOT the word "yesterday"
  "last night", "this morning", "today", "tonight" → convert to the resolved absolute date
- For events: set occurred_start AND occurred_end (same for point events)
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user.{examples}"""

# ══════════════════════════════════════════════════════════════════════════
# 选择性（极其关键）— 减少 90% 无用输出
# ══════════════════════════════════════════════════════════════════════════
#
# 只提取以下类型的事实：
#
# ✅ 个人信息：姓名、关系、角色、背景
# ✅ 偏好：喜欢、不喜欢、习惯、兴趣（例如：“Alice 喜欢咖啡”）
# ✅ 重要事件：里程碑、决策、成就、变化
# ✅ 计划 / 目标：未来意图、截止日期、承诺
# ✅ 专业能力：技能、知识、证书、经验
# ✅ 重要上下文：项目、问题、约束条件
# ✅ 感官 / 情绪细节：提供上下文的感受、知觉、体验
# ✅ 观察：对人、地点、事物的具体描述
#
# 不要提取以下内容：
#
# ❌ 泛泛问候："how are you"、"hello" 等无实际信息的寒暄
# ❌ 纯填充内容："thanks"、"sounds good"、"ok"、"got it"、"sure"
# ❌ 过程性废话："let me check"、"one moment"、"I'll look into it"
# ❌ 重复信息：已经提过的不要再次提取
#
# 尽可能将相关内容合并为一个事实。
# Concise mode guidelines
_CONCISE_GUIDELINES: str = """══════════════════════════════════════════════════════════════════════════
SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)
══════════════════════════════════════════════════════════════════════════

ONLY extract facts that are:
✅ Personal info: names, relationships, roles, background
✅ Preferences: likes, dislikes, habits, interests (e.g., "Alice likes coffee")
✅ Significant events: milestones, decisions, achievements, changes
✅ Plans/goals: future intentions, deadlines, commitments
✅ Expertise: skills, knowledge, certifications, experience
✅ Important context: projects, problems, constraints
✅ Sensory/emotional details: feelings, sensations, perceptions that provide context
✅ Observations: descriptions of people, places, things with specific details

DO NOT extract:
❌ Generic greetings: "how are you", "hello", pleasantries without substance
❌ Pure filler: "thanks", "sounds good", "ok", "got it", "sure"
❌ Process chatter: "let me check", "one moment", "I'll look into it"
❌ Repeated info: if already stated, don't extract again

CONSOLIDATE related statements into ONE fact when possible."""

# 中文翻译：
# concise 模式强调“选择性提取”，目标是尽量减少无意义输出。
# 应该提取的包括：个人信息、偏好习惯、重要事件、计划目标、技能经验、项目问题、
# 以及对长期记忆有价值的感受、观察、感官细节。
# 不应该提取的包括：打招呼、寒暄、纯 filler、过程性话术、重复信息。
# 能合并的相关内容，应尽量合并成一条 fact。

# Concise mode examples
# ══════════════════════════════════════════════════════════════════════════
# 示例（以英文展示；如果输入是非英文，所有输出必须使用输入语言）
# ══════════════════════════════════════════════════════════════════════════
#
# 示例 1 - 选择性提取（事件日期：2024年6月10日）：
#
# 输入：
# "Hey! How's it going? Good morning!
# So I'm planning my wedding - want a small outdoor ceremony.
# Just got back from Emily's wedding, she married Sarah at a rooftop garden.
# It was nice weather. I grabbed a coffee on the way."
#
# 输出：只提取 2 条事实（跳过寒暄、天气、咖啡）：
#
# 1.
# what="用户正在筹备婚礼，希望是一个小型户外仪式"
# who="user"
# why="N/A"
# entities=["user", "wedding"]
#
# 2.
# what="Emily 和 Sarah 在屋顶花园结婚"
# who="Emily（用户的朋友）, Sarah"
# occurred_start="2024-06-09"
# entities=["Emily", "Sarah", "wedding"]
#
# --------------------------------------------------
#
# 示例 2 - 职业信息：
#
# 输入：
# "Alice has 5 years of Kubernetes experience and holds CKA certification.
# She's been leading the infrastructure team since March.
# By the way, she prefers dark roast coffee."
#
# 输出：只提取 2 条事实（跳过咖啡偏好——过于琐碎）：
#
# 1.
# what="Alice 有5年 Kubernetes 经验，并持有 CKA 认证"
# who="Alice"
# entities=["Alice", "Kubernetes", "CKA"]
#
# 2.
# what="Alice 从三月起担任基础设施团队负责人"
# who="Alice"
# entities=["Alice", "infrastructure"]
#
# ══════════════════════════════════════════════════════════════════════════
# 质量优于数量
# ══════════════════════════════════════════════════════════════════════════
#
# 自问：
# “这条信息在6个月后回忆起来是否有价值？”
# 如果答案是否，则跳过。
#
# 重要说明：
# 能够为体验提供“有意义上下文”的感官 / 情绪细节和观察是值得保留的，
# 即使它们看起来很小（例如：食物的味道、一个人的外貌、音乐有多吵）。
# 如果这些信息能够刻画一次经历或一个人，就应该提取。
_CONCISE_EXAMPLES: str = """

══════════════════════════════════════════════════════════════════════════
EXAMPLES (shown in English for illustration; for non-English input, ALL output values MUST be in the input language)
══════════════════════════════════════════════════════════════════════════

Example 1 - Selective extraction (Event Date: June 10, 2024):
Input: "Hey! How's it going? Good morning! So I'm planning my wedding - want a small outdoor ceremony. Just got back from Emily's wedding, she married Sarah at a rooftop garden. It was nice weather. I grabbed a coffee on the way."

Output: ONLY 2 facts (skip greetings, weather, coffee):
1. what="User planning wedding, wants small outdoor ceremony", who="user", why="N/A", entities=["user", "wedding"]
2. what="Emily married Sarah at rooftop garden", who="Emily (user's friend), Sarah", occurred_start="2024-06-09", entities=["Emily", "Sarah", "wedding"]

Example 2 - Professional context:
Input: "Alice has 5 years of Kubernetes experience and holds CKA certification. She's been leading the infrastructure team since March. By the way, she prefers dark roast coffee."

Output: ONLY 2 facts (skip coffee preference - too trivial):
1. what="Alice has 5 years Kubernetes experience, CKA certified", who="Alice", entities=["Alice", "Kubernetes", "CKA"]
2. what="Alice leads infrastructure team since March", who="Alice", entities=["Alice", "infrastructure"]

══════════════════════════════════════════════════════════════════════════
QUALITY OVER QUANTITY
══════════════════════════════════════════════════════════════════════════

Ask: "Would this be useful to recall in 6 months?" If no, skip it.

IMPORTANT: Sensory/emotional details and observations that provide meaningful context
about experiences ARE important to remember, even if they seem small (e.g., how food
tasted, how someone looked, how loud music was). Extract these if they characterize
an experience or person."""

# Assembled concise prompt
# 中文翻译：
# 这是最终的 concise prompt，由基础 prompt、concise 规则、concise 示例三部分拼接而成。
CONCISE_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines=_CONCISE_GUIDELINES,
    examples=_CONCISE_EXAMPLES,
)

# Custom prompt uses same base but without examples
# 中文翻译：
# custom 模式仍然复用基础 prompt，
# 只是把 `{custom_instructions}` 放到规则位置，并且不带 examples。
CUSTOM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines="{custom_instructions}",
    examples="",  # No examples for custom mode
)
# ══════════════════════════════════════════════════════════════════════════
# 逐字模式（VERBATIM MODE）——只提取元数据
# ══════════════════════════════════════════════════════════════════════════
#
# 原始文本将以原样形式存储在代码中。你的唯一任务是提取元数据。
#
# 规则：
# - 对每一个输入 chunk，必须且仅生成一条记录（EXACTLY ONE）。
# - 不要包含 "what" 字段 —— 它不属于输出 schema。
# - 提取所有实体（人物、地点、组织、物体、概念）。
# - 提取时间信息（occurred_start, occurred_end, fact_kind, when）。
# - 提取地点（where）和人物（who）。
# - fact_type：默认使用 "world"，除非内容明确是与 assistant 的交互。
# Verbatim mode: preserve the original text exactly, but still extract metadata
_VERBATIM_GUIDELINES = """══════════════════════════════════════════════════════════════════════════
VERBATIM MODE — Extract metadata only
══════════════════════════════════════════════════════════════════════════

The original text will be stored as-is in code. Your ONLY job is to extract metadata.

RULES:
- Produce EXACTLY ONE entry per input chunk.
- DO NOT include a "what" field — it is not part of the output schema.
- Extract all entities (people, places, organizations, objects, concepts).
- Extract temporal information (occurred_start, occurred_end, fact_kind, when).
- Extract location (where) and people (who).
- fact_type: use "world" unless the content is clearly an interaction with the assistant."""

# 中文翻译：
# 这是最终的 verbatim prompt，由基础 prompt 加上 verbatim 专用规则组成，不带 examples。
VERBATIM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines=_VERBATIM_GUIDELINES,
    examples="",
)


# Verbose extraction prompt - detailed, comprehensive facts (legacy mode)
# 中文翻译：
# 从文本中提取事实并转换为结构化格式，包含“五个必填维度”——必须极其详细。
#
# 语言要求（强制）：
# 必须检测输入文本的语言，并且输出的所有内容必须使用完全相同的语言。
# 严格禁止翻译或切换语言。输出中的每一个词都必须与输入语言一致。
# 在任何情况下都不允许使用不同语言输出。
#
# {retain_mission_section}
#
# ══════════════════════════════════════════════════════════════════════════
# 事实格式 —— 五个维度全部必填 —— 最大信息量
# ══════════════════════════════════════════════════════════════════════════
#
# 对于每一个事实，必须捕获所有细节 —— 绝不允许总结或省略：
#
# 1. what（发生了什么）：
#    - 完整描述发生的事情
#    - 包含所有细节（对象、动作、数量、具体信息）
#
# 2. when（何时发生）：
#    - 必须包含时间信息，并包含“星期几”
#    - 格式：星期 + 月 日 年
#      例如："Monday, June 10, 2024"
#    - 星期必须写：Monday / Tuesday / ... / Sunday
#
# 3. where（发生地点）：
#    - 具体地点、区域、位置（如果有）
#
# 4. who（参与者）：
#    - 所有人物/实体
#    - 必须包含完整关系和背景信息
#
# 5. why（为什么重要）：
#    - 包含所有情绪、动机、偏好、意义、细节
#    - 对于 assistant 类型：
#      必须包含用户的请求或触发原因
#
# 额外字段：
# - fact_type
# - fact_kind
# - entities
# - occurred_start / occurred_end（结构化时间）
# - where（结构化位置）
#
# 信息密度要求：
# 必须包含所有提到的细节。细节越多越好，绝不能减少。
#
# ══════════════════════════════════════════════════════════════════════════
# 指代消解（极其重要）
# ══════════════════════════════════════════════════════════════════════════
#
# 如果文本中同时出现“关系描述 + 名字”，必须进行关联！
#
# 示例输入：
# "I went to my college roommate's wedding last June.
#  Emily finally married Sarah after 5 years together."
#
# 正确输出：
# - what: Emily 与 Sarah 在屋顶花园举行婚礼
# - when: Saturday, June 8, 2024（交往5年后）
# - where: 旧金山市中心屋顶花园
# - who: Emily（用户大学室友）, Sarah（Emily交往5年的伴侣）
# - why: 用户觉得浪漫，希望类似婚礼
# - where（结构化）: San Francisco
#
# 错误输出：
# - what: 用户室友结婚 ← 丢失名字 ❌
# - who: 室友 ← 错误 ❌
# - where: 缺失 ← 错误 ❌
#
# ══════════════════════════════════════════════════════════════════════════
# fact_kind 分类（决定时间处理方式，极其关键）
# ══════════════════════════════════════════════════════════════════════════
#
# ⚠️ 必须正确设置 fact_kind
#
# fact_kind = "event"（事件）：
# 用于：
# - 有具体时间的行为（去、买、访问等）
# - 过去事件
# - 有时间的未来计划
#
# 例：
# "I went to a workshop" → event
# "Alice visited Paris" → event
# "I bought a car yesterday" → event
#
# ------------------------------------
#
# fact_kind = "conversation"（状态）：
# 用于：
# - 持续状态（工作、居住）
# - 偏好（喜欢、讨厌）
# - 能力（会什么）
#
# 例：
# "I love Italian food" → conversation
# "Alice works at Google" → conversation
#
# ══════════════════════════════════════════════════════════════════════════
# 时间处理（极其关键）
# ══════════════════════════════════════════════════════════════════════════
#
# ⚠️ 必须使用输入中的“Event Date”作为参考时间
#
# 所有相对时间（yesterday / last week）：
# → 必须转换为绝对时间（基于 Event Date）
#
# ------------------------------------
#
# 对于 event：
#
# - 必须设置 occurred_start 和 occurred_end
# - 相对时间必须转绝对时间
# - 文中出现的时间必须提取（不能用当前时间）
# - when 字段必须带星期
# - 单点事件：start = end
#
# ------------------------------------
#
# 对于 conversation：
#
# - 不设置时间
# - 因为是长期状态
#
# ══════════════════════════════════════════════════════════════════════════
# 事实类型（fact_type）
# ══════════════════════════════════════════════════════════════════════════
#
# - world：
#   用户生活、客观事件（即使没有对话也存在）
#
# - assistant：
#   与 assistant 的交互
#
# ⚠️ 对 assistant：
# 必须包含：
# - 用户问了什么
# - 想解决什么问题
# - 提供了什么上下文
#
# ══════════════════════════════════════════════════════════════════════════
# 实体提取（必须尽可能全）
# ══════════════════════════════════════════════════════════════════════════
#
# 必须提取：
#
# - 人名（Emily, Alice）
# - 组织（Google, MIT）
# - 地点（San Francisco, Paris）
# - 重要物体（咖啡机、新车、婚纱）
# - 抽象概念（友情、职业发展）
#
# 如果事实与用户有关：
# 必须包含 "user"
#
# 任何可以帮助“关联事实”的东西都要提取。
VERBOSE_FACT_EXTRACTION_PROMPT = """Extract facts from text into structured format with FIVE required dimensions - BE EXTREMELY DETAILED.

LANGUAGE: MANDATORY — Detect the language of the input text and produce ALL output in that EXACT same language. You are STRICTLY FORBIDDEN from translating or switching to any other language. Every single word of your output must be in the same language as the input. Do NOT output in a different language under any circumstance.

{retain_mission_section}══════════════════════════════════════════════════════════════════════════
FACT FORMAT - ALL FIVE DIMENSIONS REQUIRED - MAXIMUM VERBOSITY
══════════════════════════════════════════════════════════════════════════

For EACH fact, CAPTURE ALL DETAILS - NEVER SUMMARIZE OR OMIT:

1. **what**: WHAT happened - COMPLETE description with ALL specifics (objects, actions, quantities, details)
2. **when**: WHEN it happened - ALWAYS include temporal info with DAY OF WEEK (e.g., "Monday, June 10, 2024")
   - Always include the day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
   - Format: "day_name, month day, year" (e.g., "Saturday, June 9, 2024")
3. **where**: WHERE it happened or is about - SPECIFIC locations, places, areas, regions (if applicable)
4. **who**: WHO is involved - ALL people/entities with FULL relationships and background
5. **why**: WHY it matters - ALL emotions, preferences, motivations, significance, nuance
   - For assistant facts: MUST include what the user asked/requested that triggered this!

Plus: fact_type, fact_kind, entities, occurred_start/end (for structured dates), where (structured location)

VERBOSITY REQUIREMENT: Include EVERY detail mentioned. More detail is ALWAYS better than less.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION (CRITICAL)
══════════════════════════════════════════════════════════════════════════

When text uses BOTH a generic relation AND a name for the same person → LINK THEM!

Example input: "I went to my college roommate's wedding last June. Emily finally married Sarah after 5 years together."

CORRECT output:
- what: "Emily got married to Sarah at a rooftop garden ceremony"
- when: "Saturday, June 8, 2024, after dating for 5 years"
- where: "downtown San Francisco, at a rooftop garden venue"
- who: "Emily (user's college roommate), Sarah (Emily's partner of 5 years)"
- why: "User found it romantic and beautiful, dreams of similar outdoor ceremony"
- where (structured): "San Francisco"

WRONG output:
- what: "User's roommate got married" ← LOSES THE NAME!
- who: "the roommate" ← WRONG - use the actual name!
- where: (missing) ← WRONG - include the location!

══════════════════════════════════════════════════════════════════════════
FACT_KIND CLASSIFICATION (CRITICAL FOR TEMPORAL HANDLING)
══════════════════════════════════════════════════════════════════════════

⚠️ MUST set fact_kind correctly - this determines whether occurred_start/end are set!

fact_kind="event" - USE FOR:
- Actions that happened at a specific time: "went to", "attended", "visited", "bought", "made"
- Past events: "yesterday I...", "last week...", "in March 2020..."
- Future plans with dates: "will go to", "scheduled for"
- Examples: "I went to a pottery workshop" → event
           "Alice visited Paris in February" → event
           "I bought a new car yesterday" → event
           "The user graduated from MIT in March 2020" → event

fact_kind="conversation" - USE FOR:
- Ongoing states: "works as", "lives in", "is married to"
- Preferences: "loves", "prefers", "enjoys"
- Traits/abilities: "speaks fluent French", "knows Python"
- Examples: "I love Italian food" → conversation
           "Alice works at Google" → conversation
           "I prefer outdoor dining" → conversation

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING (CRITICAL - USE EVENT DATE AS REFERENCE)
══════════════════════════════════════════════════════════════════════════

⚠️ IMPORTANT: Use the "Event Date" provided in the input as your reference point!
All relative dates ("yesterday", "last week", "recently") must be resolved relative to the Event Date, NOT today's date.

For EVENTS (fact_kind="event") - MUST SET BOTH occurred_start AND occurred_end:
- Convert relative dates → absolute using Event Date as reference
- If Event Date is "Saturday, March 15, 2020", then "yesterday" = Friday, March 14, 2020
- Dates mentioned in text (e.g., "in March 2020") should use THAT year, not current year
- CRITICAL: If the content mentions an absolute date (e.g., "March 15, 2024", "2024-03-15"), you MUST extract it and set occurred_start in ISO format
- Always include the day name (Monday, Tuesday, etc.) in the 'when' field
- Set occurred_start AND occurred_end to WHEN IT HAPPENED (not when mentioned)
- For single-day/point events: set occurred_end = occurred_start (same timestamp)

For CONVERSATIONS (fact_kind="conversation"):
- General info, preferences, ongoing states → NO occurred dates
- Examples: "loves coffee", "works as engineer"

══════════════════════════════════════════════════════════════════════════
FACT TYPE
══════════════════════════════════════════════════════════════════════════

- **world**: User's life, other people, events (would exist without this conversation)
- **assistant**: Interactions with assistant (requests, recommendations, help)
  ⚠️ CRITICAL for assistant facts: ALWAYS capture the user's request/question in the fact!
  Include: what the user asked, what problem they wanted solved, what context they provided

══════════════════════════════════════════════════════════════════════════
ENTITIES - EXTRACT EVERYTHING
══════════════════════════════════════════════════════════════════════════

Extract ALL of the following from the fact:
- People names (Emily, Alice, Dr. Smith)
- Organizations (Google, MIT, local coffee shop)
- Places (San Francisco, Brooklyn, Paris)
- Significant objects mentioned (coffee maker, new car, wedding dress)
- Abstract concepts/themes (friendship, career growth, loss, celebration)

ALWAYS include "user" when fact is about the user.
Extract anything that could help link related facts together."""


# Causal relationships section - appended when causal extraction is enabled
# 中文翻译：
# ══════════════════════════════════════════════════════════════════════════
# 因果关系（CAUSAL RELATIONSHIPS）
# ══════════════════════════════════════════════════════════════════════════
#
# 使用 causal_relations 将事实之间建立因果关联（每个事实最多 2 条）。
#
# target_index 必须小于当前事实的 index（只能指向之前的事实）。
#
# 关系类型：
# - "caused_by"（当前事实是由目标事实引起的）
#
# 示例：
# "失业 → 无法支付房租 → 搬家"
#
# - Fact 0：失业，causal_relations: null
# - Fact 1：无法支付房租，
#   causal_relations: [{target_index: 0, relation_type: "caused_by"}]
# - Fact 2：搬家，
#   causal_relations: [{target_index: 1, relation_type: "caused_by"}]
CAUSAL_RELATIONSHIPS_SECTION = """

══════════════════════════════════════════════════════════════════════════
CAUSAL RELATIONSHIPS
══════════════════════════════════════════════════════════════════════════

Link facts with causal_relations (max 2 per fact). target_index must be < this fact's index.
Type: "caused_by" (this fact was caused by the target fact)

Example: "Lost job → couldn't pay rent → moved apartment"
- Fact 0: Lost job, causal_relations: null
- Fact 1: Couldn't pay rent, causal_relations: [{target_index: 0, relation_type: "caused_by"}]
- Fact 2: Moved apartment, causal_relations: [{target_index: 1, relation_type: "caused_by"}]"""


def _build_labels_prompt_section(labels_cfg: EntityLabelsConfig | list | None, free_form_entities: bool = True) -> str:
    """Build the entity labels classification section for the extraction prompt."""
    # 中文翻译：
    # 这个函数不是一个固定常量 prompt，而是运行时动态生成一段 labels 说明。
    # 它会告诉模型：
    # - 每条 fact 都要填写 `labels` 对象
    # - 每个字段对应一个 label group
    # - 该 group 是自由文本、单值还是多值
    # - 如果允许 free-form entities，就继续正常提取 `entities`
    # - 如果是 labels-only 模式，就不要再输出普通命名实体
    if labels_cfg is None:
        return ""

    # Accept raw list for backwards compatibility
    if isinstance(labels_cfg, list):
        if not labels_cfg:
            return ""
        labels_cfg = parse_entity_labels(labels_cfg)
        if labels_cfg is None:
            return ""

    if not labels_cfg.attributes:
        return ""

    if free_form_entities:
        entities_instruction = "Classify each fact using the structured 'labels' field below. Continue extracting regular named entities in the 'entities' field."
    else:
        entities_instruction = "Classify each fact using the structured 'labels' field below. Do NOT add regular named entities — labels-only mode."

    lines = [
        "\n\n══════════════════════════════════════════════════════════════════════════",
        "ENTITY LABELS - CLASSIFICATION ATTRIBUTES",
        "══════════════════════════════════════════════════════════════════════════",
        "",
        entities_instruction,
        "",
        "For each fact, fill the 'labels' object. Each field is a label group:",
        "",
    ]

    for attr in labels_cfg.attributes:
        if attr.type == "text":
            # Free-text: no predefined values — LLM writes any relevant string or null
            lines.append(f"- {attr.key} (free text or null): {attr.description}")
        else:
            mode = "multi-value (list)" if attr.type == "multi-values" else "single value or null"
            lines.append(f"- {attr.key} ({mode}): {attr.description}")
            for v in attr.values:
                desc = f" — {v.description}" if v.description else ""
                lines.append(f'    • "{v.value}"{desc}')
        lines.append("")

    lines.append("Only assign labels when clearly applicable. Leave null/empty if the fact does not match.")
    return "\n".join(lines)


def _build_extraction_prompt_and_schema(config) -> tuple[str, type]:
    """
    Build extraction prompt and response schema based on config.

    When a taxonomy is configured, dynamically builds a Pydantic model with a
    typed `taxonomy_entities` field using an Enum built from valid taxonomy values.
    This enables JSON schema enforcement for structured outputs.

    Returns:
        Tuple of (prompt, response_schema)
    """
    # extraction_mode 决定基础提示词风格与基础 fact schema。
    # extraction_mode 用来决定本次应该选哪一种基础提示词模板。
    extraction_mode = config.retain_extraction_mode
    # extract_causal_links 决定是否让模型返回 fact 之间的因果关系。
    # extract_causal_links 用来决定是否启用“因果关系提取”这条附加能力。
    extract_causal_links = config.retain_extract_causal_links

    # Build retain_mission section if set - injected before the mode-specific guidelines
    # retain_mission 是 bank 级的“保留重点”，用于把提取偏好前置到 prompt 里。
    # retain_mission 是 bank 级的额外保留偏好，可能不存在，所以用 getattr 安全读取。
    retain_mission = getattr(config, "retain_mission", None)
    if retain_mission:
        # 有 retain_mission 时，构造一个独立的“FOCUS”段落，插到正式规则前面。
        retain_mission_section = (
            f"══════════════════════════════════════════════════════════════════════════\n"
            f"FOCUS — What to retain for this bank\n"
            f"══════════════════════════════════════════════════════════════════════════\n\n"
            f"{retain_mission}\n\n"
        )
    else:
        # 没有 retain_mission 时，直接使用空字符串，不额外影响 prompt。
        retain_mission_section = ""

    # Select base prompt based on extraction mode
    # 第一阶段：先根据 extraction_mode 选择“基础 prompt 模板”。
    if extraction_mode == "custom":
        if not config.retain_custom_instructions:
            # No custom instructions were provided, so custom mode falls back to concise extraction.
            # custom 模式如果没有真正提供自定义规则，就退回 concise 模式。
            base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                retain_mission_section=retain_mission_section,
            )
        else:
            # Use the custom prompt template and inject the caller-provided instructions.
            # 这里把调用方传入的 custom_instructions 注入到 custom prompt 模板中。
            base_prompt = CUSTOM_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                retain_mission_section=retain_mission_section,
                custom_instructions=config.retain_custom_instructions,
            )
    #         冗长的 / 信息很多的 / 细节极多的”
    elif extraction_mode == "verbose":
        # Verbose mode asks the model to keep much more detail per fact.
        # verbose 模式直接套用 verbose prompt，再填入 retain_mission_section。
        prompt = VERBOSE_FACT_EXTRACTION_PROMPT.format(
            retain_mission_section=retain_mission_section,
        )
    #     “逐字的 / 原封不动的”
    elif extraction_mode == "verbatim":
        # Verbatim mode extracts metadata only; raw chunk text becomes fact_text later.
        # verbatim 模式同样只需要把 retain_mission_section 填进模板里。
        prompt = VERBATIM_FACT_EXTRACTION_PROMPT.format(
            retain_mission_section=retain_mission_section,
        )
    else:
        # Default path: concise extraction prompt.
        # 其它未显式处理的模式，默认都走 concise prompt。
        base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
        prompt = base_prompt.format(
            retain_mission_section=retain_mission_section,
        )

    # Add causal relationships section if enabled
    # Verbatim mode never uses causal relations (no fact text to relate causally)
    # 第二阶段：根据模式和配置，决定基础 fact/response schema 长什么样。
    if extraction_mode == "verbatim":
        # Metadata-only schema for verbatim mode.
        # verbatim 没有真正的 fact 文本语义，因此使用专门的 metadata-only schema。
        base_fact_class = VerbatimExtractedFact
        base_response_class = VerbatimFactExtractionResponse
    elif extract_causal_links:
        # Extend both prompt and schema when causal links are enabled.
        # 开启因果关系提取时，不仅要给 prompt 加规则，还要选用带 causal_relations 的 schema。
        prompt = prompt + CAUSAL_RELATIONSHIPS_SECTION
        base_fact_class = ExtractedFactVerbose if extraction_mode == "verbose" else ExtractedFact
        base_response_class = FactExtractionResponseVerbose if extraction_mode == "verbose" else FactExtractionResponse
    else:
        # Use the no-causal schema variant when causal extraction is disabled.
        # 不提取因果关系时，使用不带 causal_relations 的 schema 版本。
        base_fact_class = ExtractedFactNoCausal
        base_response_class = FactExtractionResponseNoCausal

    # Add entity labels section if configured and build dynamic schema
    # Labels affect both the prompt text and the response schema.
    # 读取原始的 entity_labels 配置；没有配置时返回 None。
    entity_labels_raw = getattr(config, "entity_labels", None)
    # 把原始配置解析成统一结构，方便后面同时生成 prompt 和 schema。
    labels_cfg = parse_entity_labels(entity_labels_raw)
    # free_form_entities 决定 labels 之外，是否还允许普通自由实体继续存在。
    free_form_entities = getattr(config, "entities_allow_free_form", True)
    # Build the prompt appendix that explains how the model should fill the labels object.
    # 如果配置了 labels，这里会动态生成一段补充规则，告诉模型如何填写 labels 对象。
    labels_section = _build_labels_prompt_section(labels_cfg, free_form_entities)
    if labels_section:
        # Append label instructions after the base extraction guidance.
        # 把 labels 的说明拼到基础 prompt 后面。
        prompt = prompt + labels_section

    # Default to the base response schema unless labels require a dynamic variant.
    # 先把 response_schema 设成基础版本；如果下面存在 labels 配置，再动态替换。
    response_schema = base_response_class

    # 第三阶段：如果 labels 配置里真的定义了属性，就动态生成带 labels 的 schema。
    if labels_cfg and labels_cfg.attributes:
        # Build a typed Pydantic model for labels so JSON schema validation can enforce it.
        # 先根据 labels 配置创建一个专门的 LabelsModel。
        LabelsModel = build_labels_model(labels_cfg)
        if LabelsModel is not None:
            # These fields will be added on top of the base fact model.
            # dynamic_fields 表示要附加到基础 fact 模型上的新增字段。
            dynamic_fields: dict = {
                "labels": (
                    LabelsModel,
                    Field(
                        description="Classification labels for this fact. Fill each applicable field; leave others null/empty."
                    ),
                )
            }
            # 如果不允许 free-form entities，就额外覆盖 entities 字段描述。
            if not free_form_entities:
                dynamic_fields["entities"] = (
                    list[Entity] | None,
                    Field(default=None, description="Leave empty — labels-only mode"),
                )
            # Inherit parent's required fields and add 'labels' so it appears in the JSON schema
            # required array (the base class json_schema_extra overrides required entirely)
            # The base class may already define required fields through json_schema_extra.
            # 读取基础 fact class 通过 json_schema_extra 暴露的 required 配置。
            base_extra = base_fact_class.model_config.get("json_schema_extra")
            # Carry those required fields forward, then append the dynamic labels field below.
            # 把基础 required 字段保留下来，后面再额外追加 labels。
            base_required = cast(dict, base_extra).get("required", []) if isinstance(base_extra, dict) else []
            # Create a derived fact model that preserves base validation and adds labels.
            # 动态创建一个新的 fact 模型：继承基础模型，并把 dynamic_fields 挂上去。
            DynamicFact = create_model(
                "LabelsFact",
                __base__=base_fact_class,
                __config__=ConfigDict(
                    json_schema_mode="validation",
                    json_schema_extra={"required": [*base_required, "labels"]},
                ),
                **dynamic_fields,
            )
            # Wrap the derived fact model into the top-level response type used by the LLM call.
            # 再把新的 fact 模型包装成顶层 response 模型，供 response_format 使用。
            DynamicResponse = create_model("LabelsResponse", facts=(list[DynamicFact], ...))  # type: ignore[valid-type]
            response_schema = DynamicResponse

    # Return the final prompt text together with the final response schema.
    # 最终返回“完整 prompt + 对应 schema”这对组合。
    return prompt, response_schema


def _build_user_message(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """Build user message for fact extraction."""
    # 中文翻译：
    # 这条 user message 的含义是：
    # - 请从下面这段文本 chunk 中提取事实
    # - 当前 chunk 是第几个 / 总共有几个
    # - Event Date 是什么
    # - Context 是什么
    # - 如果有 metadata，也一起附上
    # - 最后给出真正要处理的 Text 正文
    from .orchestrator import parse_datetime_flexible

    sanitized_chunk = _sanitize_text(chunk)
    sanitized_context = _sanitize_text(context) if context else "none"

    if event_date is not None:
        event_date = parse_datetime_flexible(event_date)
        event_date_str = f"{event_date.strftime('%A, %B %d, %Y')} ({event_date.isoformat()})"
    else:
        event_date_str = "Unknown"

    metadata_section = ""
    if metadata:
        metadata_lines = "\n".join(f"  {k}: {v}" for k, v in metadata.items())
        metadata_section = f"\nMetadata:\n{metadata_lines}"

    return f"""Extract facts from the following text chunk.

Chunk: {chunk_index + 1}/{total_chunks}
Event Date: {event_date_str}
Context: {sanitized_context}{metadata_section}

Text:
{sanitized_chunk}"""


def _build_request_body(llm_config, config, prompt: str, user_message: str, response_schema: type) -> dict:
    """Build request body for LLM API call."""
    request_body = {
        "model": llm_config.model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
        "temperature": 0.1,
    }

    # Add max_completion_tokens if configured
    if config.retain_max_completion_tokens:
        request_body["max_completion_tokens"] = config.retain_max_completion_tokens

    # Add service_tier for OpenAI Flex Processing
    if llm_config.provider == "openai" and llm_config._provider_impl.openai_service_tier:
        request_body["service_tier"] = llm_config._provider_impl.openai_service_tier

    # Add response_format (JSON schema)
    if hasattr(response_schema, "model_json_schema"):
        schema = response_schema.model_json_schema()
        request_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "facts", "schema": schema},
        }

    return request_body


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    llm_config: "LLMConfig",
    config,
    agent_name: str = None,
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).

    Note: event_date parameter is kept for backward compatibility but not used in prompt.
    The LLM extracts temporal information from the context string instead.
    """
    # 局部导入，减少这个 helper 对模块级导入顺序的依赖。
    import logging

    # 专门捕获 provider SDK 的请求错误，便于识别“配置不兼容”这类非瞬时问题。
    from openai import BadRequestError

    # 使用当前模块名创建 logger，方便按 chunk 跟踪提取和重试行为。
    logger = logging.getLogger(__name__)

    # Build prompt and schema using helper function
    # prompt 负责说明提取规则，response_schema 约束模型返回 JSON 结构。
    prompt, response_schema = _build_extraction_prompt_and_schema(config)

    # Check config for extraction mode and causal link extraction
    # extraction_mode 决定最终 fact_text 的构造方式；
    # extract_causal_links 决定是否解析每条 fact 的前序因果关系。
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # Build user message using helper function
    # user_message 中会携带 chunk 正文、上下文、event_date 和 metadata。
    user_message = _build_user_message(chunk, chunk_index, total_chunks, event_date, context, metadata)

    # Retry logic for JSON validation errors
    # Use retain-specific overrides if set, otherwise fall back to global LLM config
    llm_max_retries = (
        config.retain_llm_max_retries if config.retain_llm_max_retries is not None else config.llm_max_retries
    )
    last_error: Exception | None = None

    usage = TokenUsage()  # Track cumulative usage across retries
    for attempt in range(llm_max_retries):
        try:
            # 本层重试参数允许被 retain 专用配置覆盖。
            initial_backoff = (
                config.retain_llm_initial_backoff
                if config.retain_llm_initial_backoff is not None
                else config.llm_initial_backoff
            )
            max_backoff = (
                config.retain_llm_max_backoff if config.retain_llm_max_backoff is not None else config.llm_max_backoff
            )

            # 先拿原始 JSON，再在 Python 中做宽松校验，
            # 这样可以更稳地处理模型偶发的小格式偏差。
            extraction_response_json, call_usage = await llm_config.call(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
                response_format=response_schema,
                scope="retain_extract_facts",
                temperature=0.1,
                max_completion_tokens=config.retain_max_completion_tokens,
                max_retries=llm_max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=True,  # Get raw JSON, we'll validate leniently
                return_usage=True,
            )
            usage = usage + call_usage  # Aggregate usage across retries

            # Lenient parsing of facts from raw JSON
            # 尽量 salvage 可用事实，而不是因为单条脏数据就整次失败。
            chunk_facts = []
            has_malformed_facts = False

            # Handle malformed LLM responses
            # 顶层返回必须是 dict；否则说明结果形状已经明显偏离协议。
            if not isinstance(extraction_response_json, dict):
                if attempt < llm_max_retries - 1:
                    logger.warning(
                        f"LLM returned non-dict JSON on attempt {attempt + 1}/{llm_max_retries}: {type(extraction_response_json).__name__}. Retrying..."
                    )
                    continue
                else:
                    # 重试耗尽后仍不是 dict，就放弃这个 chunk 的结构化结果，
                    # 但保留 usage，方便成本统计。
                    logger.warning(
                        f"LLM returned non-dict JSON after {llm_max_retries} attempts: {type(extraction_response_json).__name__}. "
                        f"Raw: {str(extraction_response_json)[:500]}"
                    )
                    return [], usage

            # 约定模型返回 {"facts": [...]}，缺失时按空列表处理。
            raw_facts = extraction_response_json.get("facts", [])

            if not raw_facts:
                logger.debug(
                    f"LLM response missing 'facts' field or returned empty list. "
                    f"Response: {extraction_response_json}. "
                    f"Input: "
                    f"date: {event_date.isoformat()}, "
                    f"context: {context if context else 'none'}, "
                    f"text: {chunk}"
                )

            for i, llm_fact in enumerate(raw_facts):
                # facts 数组里的每一项理论上都应是对象；否则说明模型返回局部结构异常。
                # Skip non-dict entries but track them for retry
                if not isinstance(llm_fact, dict):
                    logger.warning(f"Skipping non-dict fact at index {i}")
                    has_malformed_facts = True
                    continue

                # 将 "", [], {}, "N/A" 等无信息占位值统一折叠成 None。
                # Helper to get non-empty value
                def get_value(field_name):
                    value = llm_fact.get(field_name)
                    if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                        return value
                    return None

                # 优先读取当前 schema 的四维字段。
                # NEW FORMAT: what, when, who, why (all required)
                what = get_value("what")
                when = get_value("when")
                who = get_value("who")
                why = get_value("why")

                # 向后兼容旧 schema 的 factual_core 字段。
                # Fallback to old format if new fields not present
                if not what:
                    what = get_value("factual_core")
                if not what:
                    # In verbatim mode, 'what' is intentionally absent — text is backfilled from chunk
                    if extraction_mode != "verbatim":
                        logger.warning(f"Skipping fact {i}: missing 'what' field")
                        continue

                # Critical field: fact_type — "assistant" maps to "experience", everything else is "world".
                # If fact_type is unexpected, fall back to fact_kind before defaulting to "world".
                # 把模型返回的类型标准化为 retain 层使用的 experience / world。
                raw_fact_type = llm_fact.get("fact_type")
                if raw_fact_type == "assistant":
                    fact_type = "experience"
                elif raw_fact_type == "world":
                    fact_type = "world"
                else:
                    raw_fact_kind = llm_fact.get("fact_kind")
                    fact_type = "experience" if raw_fact_kind == "assistant" else "world"

                # Get fact_kind for temporal handling (but don't store it)
                # fact_kind 只用于控制时间字段处理，不直接存到最终 Fact 中。
                fact_kind = llm_fact.get("fact_kind", "conversation")
                if fact_kind not in ["conversation", "event", "other"]:
                    fact_kind = "conversation"

                # Build combined fact text from the 4 dimensions: what | when | who | why
                # In verbatim mode, leave combined_text empty — _collapse_to_verbatim backfills it
                # fact_data 用于暂存 occurred_*、entities、causal_relations 等可选字段。
                fact_data = {}
                if extraction_mode == "verbatim":
                    combined_text = ""
                else:
                    # 非 verbatim 模式下，将 what / when / who / why 折叠成单条 fact_text。
                    combined_parts = [what]

                    if when:
                        combined_parts.append(f"When: {when}")

                    if who:
                        combined_parts.append(f"Involving: {who}")

                    if why:
                        combined_parts.append(why)

                    combined_text = " | ".join(combined_parts)

                # Add temporal fields
                # For events: occurred_start/occurred_end (when the event happened)
                if fact_kind == "event":
                    # 只有 event 类型才补充 occurred_start / occurred_end。
                    occurred_start = get_value("occurred_start")
                    occurred_end = get_value("occurred_end")

                    # If LLM didn't set temporal fields, try to extract them from the fact text
                    # 如果模型没有给出发生时间，尝试根据相对时间词做一次兜底推断。
                    if not occurred_start:
                        fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                    else:
                        fact_data["occurred_start"] = occurred_start

                    # For point events: if occurred_end not set, default to occurred_start
                    # 点事件没有结束时间时，默认结束时间等于开始时间。
                    if occurred_end:
                        fact_data["occurred_end"] = occurred_end
                    elif fact_data.get("occurred_start"):
                        fact_data["occurred_end"] = fact_data["occurred_start"]

                # Add entities if present (validate as Entity objects)
                # LLM sometimes returns strings instead of {"text": "..."} format
                entities = get_value("entities")
                validated_entities = []
                if entities:
                    # 将字符串实体和对象实体统一归一化为 Entity 模型。
                    # Validate and normalize each entity
                    for ent in entities:
                        if isinstance(ent, str):
                            # Normalize string to Entity object
                            validated_entities.append(Entity(text=ent))
                        elif isinstance(ent, dict) and "text" in ent:
                            try:
                                validated_entities.append(Entity.model_validate(ent))
                            except Exception as e:
                                logger.warning(f"Invalid entity {ent}: {e}")

                # Post-process label entities from structured labels object
                # 如果启用了结构化 labels，把 labels 字段也转换成实体，便于统一检索和打标。
                entity_labels_raw = getattr(config, "entity_labels", None)
                labels_cfg = parse_entity_labels(entity_labels_raw)
                free_form_entities = getattr(config, "entities_allow_free_form", True)
                if labels_cfg and labels_cfg.attributes:
                    labels_lookup = build_labels_lookup(labels_cfg)
                    labels_data = llm_fact.get("labels") or {}
                    if isinstance(labels_data, dict):
                        # 用小写集合做去重，避免 labels 和普通 entities 重复写入。
                        existing_texts_lower = {e.text.lower() for e in validated_entities}
                        for group in labels_cfg.attributes:
                            value = labels_data.get(group.key)
                            if not value:
                                continue
                            values_list = value if isinstance(value, list) else [value]
                            for v in values_list:
                                if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                                    continue
                                label_str = f"{group.key}:{v.strip()}"
                                if group.type == "text":
                                    if label_str.lower() not in existing_texts_lower:
                                        validated_entities.append(Entity(text=label_str))
                                        existing_texts_lower.add(label_str.lower())
                                elif (
                                    label_str.lower() in labels_lookup and label_str.lower() not in existing_texts_lower
                                ):
                                    validated_entities.append(Entity(text=label_str))
                                    existing_texts_lower.add(label_str.lower())
                                else:
                                    logger.warning(f"Label '{label_str}' not in valid label values, skipping")

                    # In labels-only mode, keep only label entities
                    if not free_form_entities:
                        validated_entities = [
                            e for e in validated_entities if is_label_entity(e.text, labels_cfg, labels_lookup)
                        ]
                elif not free_form_entities:
                    # No labels but free_form disabled: clear all entities
                    validated_entities = []

                if validated_entities:
                    fact_data["entities"] = validated_entities

                # Add per-fact causal relations (only if enabled in config)
                if extract_causal_links:
                    # 因果关系只允许引用当前 chunk 内“之前已经出现”的 fact。
                    validated_relations = []
                    causal_relations_raw = get_value("causal_relations")
                    if causal_relations_raw:
                        for rel in causal_relations_raw:
                            if not isinstance(rel, dict):
                                continue
                            # New schema uses target_index
                            target_idx = rel.get("target_index")
                            relation_type = rel.get("relation_type")
                            strength = rel.get("strength", 1.0)

                            if target_idx is None or relation_type is None:
                                continue

                            # Validate: target_index must be < current fact index
                            if target_idx < 0 or target_idx >= i:
                                logger.debug(
                                    f"Invalid target_index {target_idx} for fact {i} (must be 0 to {i - 1}). Skipping."
                                )
                                continue

                            try:
                                # 这里先保留 chunk 内局部索引；更上层会再映射到全局索引。
                                validated_relations.append(
                                    CausalRelation(
                                        target_fact_index=target_idx,
                                        relation_type=relation_type,
                                        strength=strength,
                                    )
                                )
                            except Exception as e:
                                logger.debug(f"Invalid causal relation {rel}: {e}")

                    if validated_relations:
                        fact_data["causal_relations"] = validated_relations

                # Set mentioned_at to the event_date (when the conversation/document occurred),
                # or None when the caller opted into no timestamp.
                # mentioned_at 表示“这条内容被记录/提到的时间”，与 fact 自身发生时间不同。
                fact_data["mentioned_at"] = event_date.isoformat() if event_date is not None else None

                # Build Fact model instance
                # 单条 fact 校验失败时只跳过该条，并记录这轮输出存在 malformed 数据。
                try:
                    fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                    chunk_facts.append(fact)
                except Exception as e:
                    logger.error(f"Failed to create Fact model for fact {i}: {e}")
                    has_malformed_facts = True
                    continue

            # If we got malformed facts and haven't exhausted retries, try again
            # 如果坏数据比例过高，就整块重试一次，争取拿到更干净的结构化输出。
            if has_malformed_facts and len(chunk_facts) < len(raw_facts) * 0.8 and attempt < llm_max_retries - 1:
                logger.warning(
                    f"Got {len(raw_facts) - len(chunk_facts)} malformed facts out of {len(raw_facts)} on attempt {attempt + 1}/{llm_max_retries}. Retrying..."
                )
                continue

            # 当前轮次结果可接受，返回这个 chunk 提取到的 facts 和累计 usage。
            return chunk_facts, usage

        except BadRequestError as e:
            # 记录 provider 级错误，便于重试耗尽后向上抛出。
            last_error = e
            error_str = str(e).lower()

            # Check if error is related to max_tokens/completion_tokens not being supported
            # 这类错误通常不是“再试一次就好”，而是模型能力或配置不匹配。
            if any(
                keyword in error_str
                for keyword in [
                    "max_tokens",
                    "max_completion_tokens",
                    "maximum context",
                    "token limit",
                    "context length",
                ]
            ):
                # Provide helpful error message with configuration suggestions
                raise ValueError(
                    f"Model does not support the required output token limit.\n\n"
                    f"The model '{llm_config.model}' (provider: {llm_config.provider}) failed with: {e}\n\n"
                    f"You have two options to fix this:\n"
                    f"  1. Use a different model that supports at least {config.retain_max_completion_tokens} output tokens\n"
                    f"  2. Decrease HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS to a value your model supports\n"
                    f"     (current value: {config.retain_max_completion_tokens}, must be > RETAIN_CHUNK_SIZE={config.retain_chunk_size})"
                ) from e

            if "json_validate_failed" in str(e):
                # provider 侧 schema 校验失败有时具备随机性，因此在还有次数时允许重试。
                logger.warning(
                    f"          [1.3.{chunk_index + 1}] Attempt {attempt + 1}/{llm_max_retries} failed with JSON validation error: {e}"
                )
                if attempt < llm_max_retries - 1:
                    logger.info(f"          [1.3.{chunk_index + 1}] Retrying...")
                    continue
            # If it's not a JSON validation error or we're out of retries, re-raise
            raise

    # If we exhausted all retries, raise the last error or a descriptive fallback
    # 优先抛出最后一次真实异常；如果没有，则给一个兜底错误。
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Fact extraction failed after {llm_max_retries} attempts: LLM did not return valid JSON")


async def _extract_facts_with_auto_split(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    llm_config: LLMConfig,
    config,
    agent_name: str = None,
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a chunk with automatic splitting if output exceeds token limits.

    If the LLM output is too long (OutputTooLongError), this function automatically
    splits the chunk in half and processes each half recursively.

    Args:
        chunk: Text chunk to process
        chunk_index: Index of this chunk in the original list
        total_chunks: Total number of original chunks
        event_date: Reference date for temporal information
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        config: Resolved HindsightConfig for this bank
        agent_name: Optional agent name (memory owner)
        metadata: Optional document metadata key-value pairs

    Returns:
        Tuple of (facts list, token usage) extracted from the chunk (possibly from sub-chunks)
    """
    # 这里局部导入 logging，保持与文件其它位置一致，也避免模块级别额外依赖耦合。
    import logging

    # 使用当前模块名创建 logger，方便日志里定位到 fact_extraction.py。
    logger = logging.getLogger(__name__)

    try:
        # 第一优先级：先把当前 chunk 作为一个完整片段直接交给 LLM 提取。
        # 如果模型能够在输出上限内正常返回，这是成本最低、语义最完整的路径。
        return await _extract_facts_from_chunk(
            # 当前要处理的原始文本片段。
            chunk=chunk,
            # 这个 chunk 在原始 chunk 列表中的索引，用于日志和错误定位。
            chunk_index=chunk_index,
            # 原始总 chunk 数，只用于上下文展示和日志，不代表递归拆分后的子块数量。
            total_chunks=total_chunks,
            # 参考事件时间，供时间表达式解析使用。
            event_date=event_date,
            # 上下文信息，比如这段内容来自什么对话/文档。
            context=context,
            # LLM 连接配置，包含 provider、model 等。
            llm_config=llm_config,
            # 当前 bank 的 retain 配置。
            config=config,
            # 可选的 agent 名称，用于第一人称/agent 相关事实识别。
            agent_name=agent_name,
            # 可选元数据，传给下游 prompt 或提取逻辑使用。
            metadata=metadata,
        )
    except OutputTooLongError:
        # 只有在“输出过长”这个特定异常下，才触发自动切半逻辑。
        # 其它异常仍应向上抛出，由上层重试/失败处理负责。
        logger.warning(
            f"Output too long for chunk {chunk_index + 1}/{total_chunks} "
            f"({len(chunk)} chars). Splitting in half and retrying..."
        )

        # 先取一个粗略中点，作为默认切分位置。
        mid_point = len(chunk) // 2

        # 为了减少把一句话硬切断的概率，在中点附近搜索更自然的边界。
        # 搜索窗口设为 chunk 长度的 20%，即允许在中点前后一定范围内调整切点。
        search_range = int(len(chunk) * 0.2)
        # 搜索起点不能小于 0。
        search_start = max(0, mid_point - search_range)
        # 搜索终点不能超过 chunk 长度。
        search_end = min(len(chunk), mid_point + search_range)

        # 按优先级尝试寻找句末或段落边界。
        sentence_endings = [". ", "! ", "? ", "\n\n"]
        # 如果没找到更自然的边界，就退回到正中间硬切。
        best_split = mid_point

        for ending in sentence_endings:
            # 在搜索窗口中，从右向左找最近的分隔符，
            # 这样能尽量让切点贴近中点，同时落在句子或段落边界上。
            pos = chunk.rfind(ending, search_start, search_end)
            if pos != -1:
                # 切点放在分隔符之后，避免把 ". " 之类残留到下一半开头。
                best_split = pos + len(ending)
                # 找到第一个可接受边界就停止，遵循上面定义的优先级。
                break

        # 按最终切点把原 chunk 拆成两个子块，并去掉两端空白。
        first_half = chunk[:best_split].strip()
        second_half = chunk[best_split:].strip()

        logger.info(
            f"Split chunk {chunk_index + 1} into two sub-chunks: {len(first_half)} chars and {len(second_half)} chars"
        )

        # 对两个子块递归调用同一个函数。
        # 如果子块仍然因为输出过长失败，会继续再切半，直到可处理为止。
        # 两个子任务并行跑，尽量降低自动拆分带来的额外延迟。
        sub_tasks = [
            _extract_facts_with_auto_split(
                # 第一半文本
                chunk=first_half,
                # 仍沿用原始 chunk_index，表示它们都来源于同一个外层 chunk。
                chunk_index=chunk_index,
                # 总 chunk 数也保持原值，因为这里只是该 chunk 的内部递归拆分。
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                config=config,
                agent_name=agent_name,
                metadata=metadata,
            ),
            _extract_facts_with_auto_split(
                # 第二半文本
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                config=config,
                agent_name=agent_name,
                metadata=metadata,
            ),
        ]

        # 并行等待两个子块都提取完成。
        sub_results = await asyncio.gather(*sub_tasks)

        # 合并两个子块的 facts 和 token 使用量。
        # 这里不再保留“左右半边”的层次结构，直接返回一个扁平结果给上层。
        all_facts = []
        total_usage = TokenUsage()
        for sub_facts, sub_usage in sub_results:
            # 把该子块提取出的 facts 追加到总列表中。
            all_facts.extend(sub_facts)
            # 累加 token 使用量，便于上层统计真实成本。
            total_usage = total_usage + sub_usage

        logger.info(f"Successfully extracted {len(all_facts)} facts from split chunk {chunk_index + 1}")

        # 返回拆分后汇总得到的 facts 与 usage；
        # 对调用方来说，这和“直接从原 chunk 提取成功”的返回形状完全一致。
        return all_facts, total_usage


async def extract_facts_from_text(
    text: str,
    event_date: datetime | None,
    llm_config: LLMConfig,
    agent_name: str,
    config,
    context: str = "",
    metadata: dict[str, str] | None = None,
) -> tuple[list[Fact], list[tuple[str, int]], TokenUsage]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>3000 chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    If a chunk produces output that exceeds token limits (OutputTooLongError), it is
    automatically split in half and retried recursively until successful.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        llm_config: LLM configuration to use
        agent_name: Agent name (memory owner)
        config: Resolved HindsightConfig for this bank
        context: Context about the conversation/document
        metadata: Optional document metadata key-value pairs

    Returns:
        Tuple of (facts, chunks, usage) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk
        - usage: Aggregated token usage across all LLM calls

    实现原理：这是“单条文本 -> LLM facts”的核心入口。
    它会先把一段长文本切成多个 chunks，然后对每个 chunk 并行调用 LLM 抽取，
    最后把结果重新汇总。注意这里的输出还是 LLM 层的 Fact 模型，不是最终写库用的
    `ExtractedFactType`，后者要在更上层 orchestration 里再补齐索引、chunk 映射
    和上下文字段。
    """
    chunks = chunk_text(text, max_chars=config.retain_chunk_size)

    # 先记录切块规模，方便定位“文本过长导致提取慢/贵”的问题。
    total_chars = sum(len(c) for c in chunks)
    if len(chunks) > 1:
        logger.debug(
            f"[FACT_EXTRACTION] Text chunked into {len(chunks)} chunks ({total_chars:,} chars total, "
            f"chunk_size={config.retain_chunk_size:,}) - starting parallel LLM extraction"
        )

    # Per-chunk retry wrapper: each chunk gets up to MAX_CHUNK_RETRIES attempts.
    # This handles transient LLM failures (timeouts, rate limits, malformed responses)
    # without discarding the entire batch. If a chunk still fails after all retries,
    # the ENTIRE retain fails — we do not accept partial extraction.
    MAX_CHUNK_RETRIES = 3
    CHUNK_RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

    async def _extract_chunk_with_retry(chunk: str, chunk_index: int) -> tuple:
        """
        从单个 chunk 中提取 facts，并在失败时做有限次数重试。

        设计意图：
        1. 重试粒度控制在 chunk 级别，避免某一个分块的瞬时失败拖垮整个批次
        2. 仅处理“再试一次可能成功”的失败场景，例如临时超时、速率限制、偶发格式错误
        3. 如果该 chunk 在多次重试后仍失败，就把异常继续抛出，由上层决定整个 retain 失败

        返回值仍然保持和 `_extract_facts_with_auto_split()` 一致：
        `(chunk_facts, chunk_usage)`。
        """
        last_exception = None
        for attempt in range(MAX_CHUNK_RETRIES):
            try:
                # 真正的提取逻辑在这里执行。
                # 它除了调用 LLM 取 facts 之外，还会在输出过长时自动把 chunk 再切小并递归处理。
                return await _extract_facts_with_auto_split(
                    chunk=chunk,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks),
                    event_date=event_date,
                    context=context,
                    llm_config=llm_config,
                    config=config,
                    agent_name=agent_name,
                    metadata=metadata,
                )
            except Exception as e:
                # 记录最后一次异常；如果所有重试都失败，最终抛出这个异常给上层。
                last_exception = e
                if attempt < MAX_CHUNK_RETRIES - 1:
                    # 指数退避：第一次等 2 秒，第二次等 4 秒……
                    # 目的是给模型服务、网络抖动或限流留出恢复时间。
                    delay = CHUNK_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"Chunk {chunk_index}/{len(chunks)} extraction failed "
                        f"(attempt {attempt + 1}/{MAX_CHUNK_RETRIES}): "
                        f"{type(e).__name__}. Retrying in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # 走到这里说明这个 chunk 已经耗尽重试次数。
                    # 注意这里只记录日志，不吞掉异常；函数末尾会继续抛出。
                    logger.error(
                        f"Chunk {chunk_index}/{len(chunks)} extraction failed after "
                        f"{MAX_CHUNK_RETRIES} attempts: {type(e).__name__}: {e}"
                    )
        # 所有尝试都失败时，把最后一次异常交给上层。
        # 上层会据此把该 chunk 记为失败，并最终使整个 retain 失败，而不是接受部分提取结果。
        raise last_exception

    tasks = [_extract_chunk_with_retry(chunk, i) for i, chunk in enumerate(chunks)]

    # return_exceptions=True so we can collect all results even if some chunks
    # exhausted their retries. We check for failures below and fail the retain
    # if ANY chunk could not be extracted — partial extraction is not acceptable.
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_facts = []
    chunk_metadata = []  # [(chunk_text, fact_count), ...]
    total_usage = TokenUsage()
    failed_chunks = []
    for i, (chunk, result) in enumerate(zip(chunks, chunk_results)):
        if isinstance(result, Exception):
            failed_chunks.append((i, result))
            continue
        chunk_facts, chunk_usage = result
        all_facts.extend(chunk_facts)
        chunk_metadata.append((chunk, len(chunk_facts)))
        total_usage = total_usage + chunk_usage

    if failed_chunks:
        # Fail the entire retain — partial extraction is not acceptable.
        # All successfully extracted facts are discarded because the transaction
        # hasn't committed yet. The worker poller will retry the entire task.
        failed_summary = ", ".join(f"chunk {idx}: {type(err).__name__}" for idx, err in failed_chunks[:5])
        raise RuntimeError(
            f"Fact extraction failed: {len(failed_chunks)}/{len(chunks)} chunks failed "
            f"after {MAX_CHUNK_RETRIES} retries each. First failures: {failed_summary}"
        )

    return all_facts, chunk_metadata, total_usage


# ============================================================================
# ORCHESTRATION LAYER
# ============================================================================

# Import types for the orchestration layer (note: ExtractedFact here is different from the Pydantic model above)

from .types import CausalRelation as CausalRelationType
from .types import ChunkMetadata, RetainContent
from .types import ExtractedFact as ExtractedFactType

logger = logging.getLogger(__name__)

# Each fact gets 10ms offset to preserve ordering within a document
SECONDS_PER_FACT = 0.01


async def extract_facts_from_contents_batch_api(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts using LLM Batch API (OpenAI/Groq).

    Submits all chunks as a single batch, polls until complete, then processes results.
    Only called when config.retain_batch_enabled=True.

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration with batch API support
        agent_name: Name of the agent
        config: Resolved HindsightConfig for this bank
        pool: Database connection pool (for storing batch state)
        operation_id: Async operation ID (for crash recovery)
        schema: Database schema (for multi-tenant support)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)

    实现原理：这是 `extract_facts_from_contents()` 的 Batch API 分支。
    当 provider 支持批处理时，它不会逐 chunk 立即同步调用 LLM，而是：
    1. 先把所有 chunks 组装成 batch requests
    2. 一次性提交给 provider
    3. 轮询等待 batch 完成
    4. 再把批处理结果重新映射回 chunk/content 结构

    这样适合高吞吐、大批量 retain 场景，也支持把 `batch_id` 写入 operation metadata，
    让 worker 崩溃后可以恢复轮询而不是重新提交整批请求。
    """
    if not contents:
        return [], [], TokenUsage()

    logger.info(f"Using Batch API for fact extraction ({len(contents)} contents)")

    # Check config for extraction mode and causal link extraction (used throughout)
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # batch 模式是能力增强，不是强依赖；provider 不支持时要平滑回退到同步模式。
    if not await llm_config._provider_impl.supports_batch_api():
        logger.warning(f"Batch API not supported for provider {llm_config.provider}, falling back to sync mode")
        return await extract_facts_from_contents(contents, llm_config, agent_name, config, pool, operation_id, schema)

    # 如果 operation metadata 里已经有 batch_id，说明这次是 crash recovery，
    # 应当继续轮询旧 batch，而不是重复提交一份新的 batch 请求。
    batch_id = None
    if operation_id and pool:
        from ..task_backend import fq_table

        table = fq_table("async_operations", schema)
        row = await pool.fetchrow(
            f"SELECT result_metadata FROM {table} WHERE operation_id = $1",
            operation_id,
        )

        if row and row["result_metadata"]:
            metadata = row["result_metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            batch_id = metadata.get("batch_id")

            if batch_id:
                logger.info(f"Resuming existing batch: batch_id={batch_id} (crash recovery)")

    # 先把所有 content 展平成 chunk 请求队列；后续 batch 结果会按 custom_id
    # 再映射回 content_index 和 chunk_index_in_content。
    all_chunks_info = []  # List of (chunk_text, content_index, chunk_index_in_content, event_date, context)
    batch_requests = []

    # Build prompt and schema once (same for all chunks)
    prompt, response_schema = _build_extraction_prompt_and_schema(config)

    for content_index, item in enumerate(contents):
        chunks = chunk_text(item.content, max_chars=config.retain_chunk_size)

        for chunk_index_in_content, chunk in enumerate(chunks):
            all_chunks_info.append((chunk, content_index, chunk_index_in_content, item.event_date, item.context))

            # Build batch request for this chunk
            custom_id = f"chunk_{len(all_chunks_info) - 1}"  # Global chunk index

            # Build user message using helper function
            user_message = _build_user_message(
                chunk, chunk_index_in_content, len(chunks), item.event_date, item.context, item.metadata or None
            )

            # Build request body using helper function
            request_body = _build_request_body(llm_config, config, prompt, user_message, response_schema)

            batch_requests.append(
                {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
            )

    if not batch_requests and not batch_id:  # No requests and not resuming
        return [], [], TokenUsage()

    # Step 2: Submit batch (skip if resuming)
    if not batch_id:
        logger.info(f"Submitting batch with {len(batch_requests)} chunk requests")

        batch_metadata = await llm_config._provider_impl.submit_batch(batch_requests)
        batch_id = batch_metadata["batch_id"]

        logger.info(f"Batch submitted: {batch_id}, polling every {config.retain_batch_poll_interval_seconds}s")

        # CRITICAL: Store minimal batch state in operation metadata for crash recovery
        # This allows resuming polling if worker restarts
        if operation_id and pool:
            batch_state = {
                "batch_id": batch_id,
                "batch_provider": llm_config.provider,
                "chunk_count": len(batch_requests),
            }

            # Update operation result_metadata
            from ..task_backend import fq_table

            table = fq_table("async_operations", schema)
            await pool.execute(
                f"""
                UPDATE {table}
                SET result_metadata = result_metadata || $1::jsonb, updated_at = now()
                WHERE operation_id = $2
                """,
                json.dumps(batch_state),
                operation_id,
            )
            logger.info(f"Stored batch state for operation {operation_id} (crash recovery enabled)")
    else:
        logger.info(f"Resuming polling for existing batch: {batch_id}")

    # Step 3: Poll until complete
    import time

    start_time = time.time()
    while True:
        status_info = await llm_config._provider_impl.get_batch_status(batch_id)
        status = status_info["status"]

        elapsed = time.time() - start_time
        logger.info(
            f"Batch {batch_id}: status={status}, "
            f"completed={status_info['request_counts']['completed']}/{status_info['request_counts']['total']}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            break
        elif status in ("failed", "expired", "cancelled"):
            error_msg = status_info.get("errors", "Unknown error")
            raise RuntimeError(f"Batch {batch_id} failed with status {status}: {error_msg}")

        # Wait before polling again
        await asyncio.sleep(config.retain_batch_poll_interval_seconds)

    logger.info(f"Batch {batch_id} completed in {elapsed:.0f}s, retrieving results")

    # Step 4: Retrieve results
    batch_results = await llm_config._provider_impl.retrieve_batch_results(batch_id)

    # Map results by custom_id
    results_by_id = {result["custom_id"]: result for result in batch_results}

    # 批处理完成后，解析逻辑仍然尽量和同步模式保持一致，这样两条路径的语义更容易对齐。
    all_facts_from_llm = []
    chunks_metadata = []
    total_usage = TokenUsage()

    for chunk_idx, (chunk_content, content_index, chunk_index_in_content, event_date, context) in enumerate(
        all_chunks_info
    ):
        custom_id = f"chunk_{chunk_idx}"
        result = results_by_id.get(custom_id)

        if not result:
            logger.warning(f"Missing result for {custom_id}, skipping")
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Check for errors
        if result.get("error"):
            logger.error(f"Error in {custom_id}: {result['error']}")
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Extract response
        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])

        if not choices:
            logger.warning(f"No choices in response for {custom_id}")
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Parse JSON content
        message = choices[0].get("message", {})
        content_str = message.get("content", "{}")

        try:
            extraction_response_json = json.loads(content_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {custom_id}: {e}")
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Parse facts (reuse existing logic from _extract_facts_from_chunk)
        raw_facts = extraction_response_json.get("facts", [])
        chunk_facts = []

        for i, llm_fact in enumerate(raw_facts):
            if not isinstance(llm_fact, dict):
                continue

            def get_value(field_name):
                value = llm_fact.get(field_name)
                if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                    return value
                return None

            what = get_value("what")
            if not what:
                what = get_value("factual_core")
            if not what:
                continue

            when = get_value("when")
            who = get_value("who")
            why = get_value("why")

            # Critical field: fact_type — only "assistant" maps to "experience", everything else is "world"
            # Critical field: fact_type — "assistant" maps to "experience", everything else is "world".
            # If fact_type is unexpected, fall back to fact_kind before defaulting to "world".
            raw_fact_type = llm_fact.get("fact_type")
            if raw_fact_type == "assistant":
                fact_type = "experience"
            elif raw_fact_type == "world":
                fact_type = "world"
            else:
                raw_fact_kind = llm_fact.get("fact_kind")
                fact_type = "experience" if raw_fact_kind == "assistant" else "world"

            # Build combined fact text
            combined_parts = [what]
            if when:
                combined_parts.append(f"When: {when}")
            if who:
                combined_parts.append(f"Involving: {who}")
            if why:
                combined_parts.append(why)
            combined_text = " | ".join(combined_parts)

            # Temporal fields
            fact_data = {}
            fact_kind = llm_fact.get("fact_kind", "conversation")
            if fact_kind not in ["conversation", "event", "other"]:
                fact_kind = "conversation"

            if fact_kind == "event":
                occurred_start = get_value("occurred_start")
                occurred_end = get_value("occurred_end")

                if not occurred_start:
                    fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                else:
                    fact_data["occurred_start"] = occurred_start

                if occurred_end:
                    fact_data["occurred_end"] = occurred_end
                elif fact_data.get("occurred_start"):
                    fact_data["occurred_end"] = fact_data["occurred_start"]

            # Entities
            entities = get_value("entities")
            validated_entities = []
            if entities:
                for ent in entities:
                    if isinstance(ent, str):
                        validated_entities.append(Entity(text=ent))
                    elif isinstance(ent, dict) and "text" in ent:
                        try:
                            validated_entities.append(Entity.model_validate(ent))
                        except Exception:
                            pass

            # Post-process label entities from structured labels object
            entity_labels_raw = getattr(config, "entity_labels", None)
            labels_cfg_batch = parse_entity_labels(entity_labels_raw)
            free_form_entities_batch = getattr(config, "entities_allow_free_form", True)
            if labels_cfg_batch and labels_cfg_batch.attributes:
                labels_lookup_batch = build_labels_lookup(labels_cfg_batch)
                labels_data = llm_fact.get("labels") or {}
                if isinstance(labels_data, dict):
                    existing_texts_lower = {e.text.lower() for e in validated_entities}
                    for group in labels_cfg_batch.attributes:
                        value = labels_data.get(group.key)
                        if not value:
                            continue
                        values_list = value if isinstance(value, list) else [value]
                        for v in values_list:
                            if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                                continue
                            label_str = f"{group.key}:{v.strip()}"
                            if group.type == "text":
                                if label_str.lower() not in existing_texts_lower:
                                    validated_entities.append(Entity(text=label_str))
                                    existing_texts_lower.add(label_str.lower())
                            elif (
                                label_str.lower() in labels_lookup_batch
                                and label_str.lower() not in existing_texts_lower
                            ):
                                validated_entities.append(Entity(text=label_str))
                                existing_texts_lower.add(label_str.lower())

                if not free_form_entities_batch:
                    validated_entities = [
                        e for e in validated_entities if is_label_entity(e.text, labels_cfg_batch, labels_lookup_batch)
                    ]
            elif not free_form_entities_batch:
                validated_entities = []

            if validated_entities:
                fact_data["entities"] = validated_entities

            # Causal relations
            if extract_causal_links:
                validated_relations = []
                causal_relations_raw = get_value("causal_relations")
                if causal_relations_raw:
                    for rel in causal_relations_raw:
                        if not isinstance(rel, dict):
                            continue
                        target_idx = rel.get("target_index")
                        relation_type = rel.get("relation_type")
                        strength = rel.get("strength", 1.0)

                        if target_idx is None or relation_type is None:
                            continue
                        if target_idx < 0 or target_idx >= i:
                            continue

                        try:
                            validated_relations.append(
                                CausalRelation(
                                    target_fact_index=target_idx, relation_type=relation_type, strength=strength
                                )
                            )
                        except Exception:
                            pass

                if validated_relations:
                    fact_data["causal_relations"] = validated_relations

            # Set mentioned_at to the event_date (when the conversation/document occurred),
            # or None when the caller opted into no timestamp.
            fact_data["mentioned_at"] = event_date.isoformat() if event_date is not None else None

            try:
                fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                chunk_facts.append(fact)
            except Exception as e:
                logger.error(f"Failed to create Fact model for fact {i}: {e}")
                continue

        all_facts_from_llm.extend(chunk_facts)
        chunks_metadata.append(
            ChunkMetadata(
                chunk_text=chunk_content,
                fact_count=len(chunk_facts),
                content_index=content_index,
                chunk_index=chunk_idx,
            )
        )

        # Track token usage
        usage_data = response_body.get("usage", {})
        if usage_data:
            total_usage = total_usage + TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

    # Step 6: Convert to ExtractedFact objects with proper chunk mapping
    # Group facts by chunk
    facts_by_chunk = []  # List of (chunk_metadata, [facts])
    fact_start_idx = 0

    for chunk_meta in chunks_metadata:
        chunk_facts = all_facts_from_llm[fact_start_idx : fact_start_idx + chunk_meta.fact_count]
        facts_by_chunk.append((chunk_meta, chunk_facts))
        fact_start_idx += chunk_meta.fact_count

    # Now convert to ExtractedFactType
    extracted_facts = []
    global_fact_idx = 0

    for chunk_meta, chunk_facts in facts_by_chunk:
        content = contents[chunk_meta.content_index]

        for fact_from_llm in chunk_facts:
            extracted_fact = ExtractedFactType(
                fact_text=fact_from_llm.fact,
                fact_type="experience" if fact_from_llm.fact_type == "assistant" else "world",
                entities=[e.text for e in (fact_from_llm.entities or [])],
                occurred_start=_parse_datetime(fact_from_llm.occurred_start) if fact_from_llm.occurred_start else None,
                occurred_end=_parse_datetime(fact_from_llm.occurred_end) if fact_from_llm.occurred_end else None,
                causal_relations=_convert_causal_relations(fact_from_llm.causal_relations or [], global_fact_idx),
                content_index=chunk_meta.content_index,
                chunk_index=chunk_meta.chunk_index,
                context=content.context,
                mentioned_at=content.event_date,
                metadata=content.metadata,
                tags=content.tags,
                observation_scopes=content.observation_scopes,
            )

            extracted_facts.append(extracted_fact)
            global_fact_idx += 1

    # Step 7: Add temporal offsets
    _add_temporal_offsets(extracted_facts, contents)

    # Step 8: Auto-tag facts from label groups with tag=True
    _inject_label_tags(extracted_facts, config)

    logger.info(f"Batch API extracted {len(extracted_facts)} facts from {len(all_chunks_info)} chunks")

    return extracted_facts, chunks_metadata, total_usage


def _extract_facts_chunks(
    contents: list[RetainContent],
    config,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    chunks mode: no LLM call, no entity extraction.

    实现原理：这是 `retain_extraction_mode="chunks"` 时的降级路径。
    系统不再尝试抽取结构化 facts，而是把每个 chunk 原样当作一条 world fact 存下。
    这样在没有可用 LLM 时，系统仍能作为 chunk store 工作，至少保留原文切片和
    基础检索能力。

    Each chunk becomes one memory unit with the raw text as fact_text.
    User-provided entities from RetainContent.entities are picked up downstream
    by entity_processing.py — they are the sole source of entity data in this mode.
    """
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    global_chunk_idx = 0

    for content_index, content in enumerate(contents):
        chunks = chunk_text(content.content, config.retain_chunk_size)
        for chunk in chunks:
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk,
                    fact_count=1,
                    content_index=content_index,
                    chunk_index=global_chunk_idx,
                )
            )
            extracted_facts.append(
                ExtractedFactType(
                    fact_text=chunk,
                    fact_type="world",
                    entities=[],
                    content_index=content_index,
                    chunk_index=global_chunk_idx,
                    context=content.context,
                    mentioned_at=content.event_date,
                    metadata=content.metadata,
                    tags=content.tags,
                    observation_scopes=content.observation_scopes,
                )
            )
            global_chunk_idx += 1

    _add_temporal_offsets(extracted_facts, contents)
    return extracted_facts, chunks_metadata, TokenUsage()


async def extract_facts_from_contents(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts from multiple content items in parallel.

    This function:
    1. Extracts facts from all contents in parallel using the LLM
    2. Tracks which facts came from which chunks
    3. Adds time offsets to preserve fact ordering within each content
    4. Returns typed ExtractedFact and ChunkMetadata objects

    Routes to batch API mode if config.retain_batch_enabled=True.

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration for fact extraction
        agent_name: Name of the agent (for agent-related fact detection)
        config: Resolved HindsightConfig for this bank
        pool: Database connection pool (passed to batch API for state storage)
        operation_id: Async operation ID (passed to batch API for crash recovery)
        schema: Database schema (passed to batch API for multi-tenant support)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)

    实现原理：这是 retain 里“多内容项 -> 统一事实列表”的主入口。
    它不直接做数据库写入，而是专门负责把一批 `RetainContent` 变成后续可入库的
    `ExtractedFactType` 和 `ChunkMetadata`。整体流程是：
    1. 根据 extraction mode 选择路径：`chunks` / `batch_api` / 普通同步模式
    2. 如果是普通模式，对每个 content 并行调用 `extract_facts_from_text()`
    3. 把每个 content 的 chunk/fact 结果展平成全局索引
    4. 根据模式做 verbatim 收敛、时间偏移、自动标签注入

    这层函数的关键价值是“统一语义出口”：
    无论底层到底是直接 LLM 调用、Batch API，还是 chunks 降级模式，最终往上游
    都返回相同结构，方便 orchestration 层继续处理。
    """
    if not contents:
        return [], [], TokenUsage()

    # `chunks` 模式必须最先判定，因为它代表“完全不走 LLM”。
    # 只要命中这个分支，就不应该再触碰 Batch API 或任何 LLM 相关资源。
    if config.retain_extraction_mode == "chunks":
        return _extract_facts_chunks(contents, config)

    # 如果启用了 batch 并且 provider 支持，就把所有内容路由到批量提取路径。
    if config.retain_batch_enabled:
        return await extract_facts_from_contents_batch_api(
            contents, llm_config, agent_name, config, pool, operation_id, schema
        )

    # 普通模式下，以 content 为粒度并行提取。
    # 这里不是每个 content 再自己管最终写库，而只是并行拿到“文本级抽取结果”。
    fact_extraction_tasks = []
    for item in contents:
        # Call extract_facts_from_text directly (defined earlier in this file)
        # to avoid circular import with utils.extract_facts
        task = extract_facts_from_text(
            text=item.content,
            event_date=item.event_date,
            context=item.context,
            llm_config=llm_config,
            agent_name=agent_name,
            config=config,
            metadata=item.metadata or None,
        )
        fact_extraction_tasks.append(task)

    # 先等待全部内容项都返回，再统一折叠结果。
    # `return_exceptions=True` 让单个 content 失败不会直接取消兄弟任务。
    all_fact_results = await asyncio.gather(*fact_extraction_tasks, return_exceptions=True)

    # 这里开始把“每个 content 自己的本地结果”折叠成整个批次的全局结果。
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    total_usage = TokenUsage()

    global_chunk_idx = 0
    global_fact_idx = 0

    # 单个 content 提取失败时，不让它污染其它内容项；这里把它降级成空结果，
    # 这样上层仍可继续处理成功的那些 content。
    valid_results = []
    for content, result in zip(contents, all_fact_results):
        if isinstance(result, Exception):
            logger.warning(f"Content extraction failed (skipping): {type(result).__name__}: {result}")
            valid_results.append((content, ([], [], TokenUsage())))
        else:
            valid_results.append((content, result))

    for content_index, (content, (facts_from_llm, chunks_from_llm, content_usage)) in enumerate(valid_results):
        total_usage = total_usage + content_usage
        chunk_start_idx = global_chunk_idx

        # 先建立 chunk 的全局索引。后面 facts 会通过 `chunk_index` 指回这些 chunks。
        for chunk_index_in_content, (chunk_text, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_metadata = ChunkMetadata(
                chunk_text=chunk_text,
                fact_count=chunk_fact_count,
                content_index=content_index,
                chunk_index=global_chunk_idx,
            )
            chunks_metadata.append(chunk_metadata)
            global_chunk_idx += 1

        # 再把 LLM 的 Fact 模型补齐成 retain 层统一使用的 `ExtractedFactType`，
        # 并把 content/chunk/global fact 索引全部固定下来。
        fact_idx_in_content = 0
        for chunk_idx_in_content, (chunk_text, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_global_idx = chunk_start_idx + chunk_idx_in_content

            for _ in range(chunk_fact_count):
                if fact_idx_in_content < len(facts_from_llm):
                    fact_from_llm = facts_from_llm[fact_idx_in_content]

                    # `mentioned_at` 始终来自原 content 的 event_date；而发生时间
                    # `occurred_start/end` 则由 LLM 按具体 fact 内容判断。
                    extracted_fact = ExtractedFactType(
                        fact_text=fact_from_llm.fact,
                        fact_type="experience" if fact_from_llm.fact_type == "assistant" else "world",
                        entities=[e.text for e in (fact_from_llm.entities or [])],
                        # occurred_start/end: from LLM only, leave None if not provided
                        occurred_start=_parse_datetime(fact_from_llm.occurred_start)
                        if fact_from_llm.occurred_start
                        else None,
                        occurred_end=_parse_datetime(fact_from_llm.occurred_end)
                        if fact_from_llm.occurred_end
                        else None,
                        causal_relations=_convert_causal_relations(
                            fact_from_llm.causal_relations or [], global_fact_idx
                        ),
                        content_index=content_index,
                        chunk_index=chunk_global_idx,
                        context=content.context,
                        # mentioned_at: always the event_date (when the conversation/document occurred)
                        mentioned_at=content.event_date,
                        metadata=content.metadata,
                        tags=content.tags,
                        observation_scopes=content.observation_scopes,
                    )

                    extracted_facts.append(extracted_fact)
                    global_fact_idx += 1
                    fact_idx_in_content += 1

    # verbatim 模式要求“一块一个 fact，且 fact_text 就是原始 chunk 文本”。
    # 即使 LLM 误返回多个 fact，这里也要收敛成 chunk 粒度。
    if config.retain_extraction_mode == "verbatim":
        extracted_facts = _collapse_to_verbatim(extracted_facts, chunks_metadata)

    # 给所有时间字段加细粒度偏移，避免同批 facts 拥有完全相同的时间戳，导致
    # 检索和排序阶段丢失原始先后顺序。
    _add_temporal_offsets(extracted_facts, contents)

    # 根据 label groups 自动补 tags，把实体标签策略下沉到事实级数据。
    _inject_label_tags(extracted_facts, config)

    return extracted_facts, chunks_metadata, total_usage


def _collapse_to_verbatim(facts: list[ExtractedFactType], chunks: list[ChunkMetadata]) -> list[ExtractedFactType]:
    """
    For verbatim mode: ensure one fact per chunk with the original chunk text preserved.

    The LLM prompt asks for exactly one fact per chunk, but if it returns more,
    this collapses them: keeps the first fact as representative, overrides its
    fact_text with the raw chunk text, and merges entities from any extra facts.

    实现原理：verbatim 模式并不信任 LLM 一定严格遵守“一块一个 fact”的约束，
    所以这里做最后一道收敛：
    - 每个 chunk 只保留一个代表 fact
    - 代表 fact 的 `fact_text` 被强制改回原始 chunk 文本
    - 额外返回的 facts 只贡献实体，不再作为独立 fact 保留
    """
    chunk_text_map = {c.chunk_index: c.chunk_text for c in chunks}
    seen: dict[int, ExtractedFactType] = {}
    result: list[ExtractedFactType] = []

    for fact in facts:
        if fact.chunk_index not in seen:
            fact.fact_text = chunk_text_map.get(fact.chunk_index, fact.fact_text)
            seen[fact.chunk_index] = fact
            result.append(fact)
        else:
            # Merge entities from extra facts into the representative
            representative = seen[fact.chunk_index]
            for entity in fact.entities:
                if entity not in representative.entities:
                    representative.entities.append(entity)

    return result


def _parse_datetime(date_str: str):
    """Parse ISO datetime string."""
    from dateutil import parser as date_parser

    try:
        return date_parser.isoparse(date_str)
    except Exception:
        return None


def _convert_causal_relations(relations_from_llm, fact_start_idx: int) -> list[CausalRelationType]:
    """
    Convert causal relations from LLM format to ExtractedFact format.

    Adjusts target_fact_index from content-relative to global indices.
    """
    causal_relations = []
    for rel in relations_from_llm:
        causal_relation = CausalRelationType(
            relation_type=rel.relation_type,
            target_fact_index=fact_start_idx + rel.target_fact_index,
            strength=rel.strength,
        )
        causal_relations.append(causal_relation)
    return causal_relations


def _add_temporal_offsets(facts: list[ExtractedFactType], contents: list[RetainContent]) -> None:
    """
    Add time offsets to preserve fact ordering across all contents.

    This allows retrieval to distinguish between facts from different documents/conversations
    even when they have the same base event_date, and also between facts within the same
    conversation.

    Uses absolute position across all facts to ensure unique timestamps.

    Modifies facts in place.

    实现原理：很多 facts 共享同一个 `event_date` 或 `mentioned_at`。如果完全保留
    原值，后续按时间排序时会出现大量并列，丢失“原文中先后出现的顺序”。
    因此这里按全局 fact 顺序为时间字段加一个极小偏移量，既不破坏原始日期语义，
    又能稳定保留事实顺序。
    """
    from .orchestrator import parse_datetime_flexible

    for i, fact in enumerate(facts):
        # Use absolute position across all facts to ensure uniqueness across different contents
        offset = timedelta(seconds=i * SECONDS_PER_FACT)

        # Apply offset to all temporal fields (handle both datetime objects and ISO strings)
        if fact.occurred_start:
            fact.occurred_start = parse_datetime_flexible(fact.occurred_start) + offset
        if fact.occurred_end:
            fact.occurred_end = parse_datetime_flexible(fact.occurred_end) + offset
        if fact.mentioned_at:
            fact.mentioned_at = parse_datetime_flexible(fact.mentioned_at) + offset


def _inject_label_tags(facts: list[ExtractedFactType], config) -> None:
    """
    For label groups with tag=True, add extracted key:value label entities
    to each fact's tags list. Modifies facts in place.

    This lets entity labels double as tags, enabling filtering via the
    existing tags API without any extra query infrastructure.
    """
    labels_cfg = parse_entity_labels(getattr(config, "entity_labels", None))
    if not labels_cfg:
        return
    tag_group_keys = {g.key.lower() for g in labels_cfg.attributes if g.tag}
    if not tag_group_keys:
        return
    for fact in facts:
        label_tags = [e for e in fact.entities if ":" in e and e.split(":", 1)[0].lower() in tag_group_keys]
        if label_tags:
            existing = set(fact.tags)
            fact.tags = fact.tags + [t for t in label_tags if t not in existing]
