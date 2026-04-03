"""
Entity processing for retain pipeline.

Handles entity extraction, resolution, and link creation for stored facts.
"""

import logging

from . import link_utils
from .types import EntityLink, ProcessedFact

logger = logging.getLogger(__name__)


def _prepare_facts_for_entity_processing(
    facts: list[ProcessedFact],
    user_entities_per_content: dict[int, list[dict]] | None = None,
) -> tuple[list[str], list, list[list[dict]]]:
    """
    Extract fact texts, dates, and merged entity lists from ProcessedFact objects.

    实现原理：
    1. `resolve_entities` 后续真正需要的输入，并不是完整的 `ProcessedFact`，
       而是“事实文本 + 时间 + 实体候选列表”这三类轻量数据。
    2. 这里先把 `ProcessedFact` 拆平，减少后续实体解析阶段对 retain 内部结构的耦合。
    3. 每条 fact 的实体候选，既包含 LLM 自动抽取的实体，也包含用户在原始
       content 上显式传入的实体。
    4. 两类实体会按文本做一次大小写不敏感去重，避免同一个实体被重复解析。
    5. 输出结果与 `facts` 顺序严格对齐，后续才能按下标把解析出的实体重新挂回
       对应的 unit。

    Returns:
        Tuple of (fact_texts, fact_dates, entities_per_fact)
    """
    # 上游不传时统一转为空 dict，避免后面反复判断 None。
    user_entities_per_content = user_entities_per_content or {}

    # 先抽出每条 fact 的主文本；实体解析阶段本质上是“从文本出发做实体标准化”。
    fact_texts = [fact.fact_text for fact in facts]
    # 为每条 fact 选一个代表时间。
    # 优先用 occurred_start，表示事实真正发生的时间；没有的话退回 mentioned_at，
    # 这样实体解析/消歧仍然可以利用时间线索。
    fact_dates = [fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at for fact in facts]

    # 与 facts 一一对应的实体候选列表。
    entities_per_fact = []
    for fact in facts:
        # 先把 LLM 抽取出的实体转成统一 dict 结构，便于后续和用户实体合并。
        # 这里统一标成 CONCEPT，是因为当前这层只负责“提供候选实体”，
        # 更细的 taxonomy 由后续 resolver/labels 体系决定。
        llm_entities = [{"text": entity.name, "type": "CONCEPT"} for entity in (fact.entities or [])]

        # 用户显式传入的 entities 是按 content 维度提供的，因此要用 fact.content_index
        # 找到当前 fact 所属原始 content 的那组实体。
        user_entities = user_entities_per_content.get(fact.content_index, [])

        # 用小写文本做去重键，避免 "OpenAI" / "openai" 这类大小写差异造成重复解析。
        seen_texts = {e["text"].lower() for e in llm_entities}
        for user_entity in user_entities:
            # 只有当这条用户实体还没被 LLM 实体覆盖时，才补充进去。
            if user_entity["text"].lower() not in seen_texts:
                llm_entities.append(
                    {
                        "text": user_entity["text"],
                        "type": user_entity.get("type", "CONCEPT"),
                    }
                )
                # 立即登记到去重集合，防止同一 content 内用户重复传同名实体。
                seen_texts.add(user_entity["text"].lower())

        # 保持和 facts 同顺序追加，后面 resolver 返回的映射会依赖这个位置关系。
        entities_per_fact.append(llm_entities)

    return fact_texts, fact_dates, entities_per_fact


async def resolve_entities(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    facts: list[ProcessedFact],
    log_buffer: list[str] = None,
    user_entities_per_content: dict[int, list[dict]] = None,
    entity_labels: list | None = None,
) -> tuple[list[str], list[tuple], dict[str, list[str]]]:
    """
    Phase 1: Resolve entity names to canonical IDs (read-heavy).

    Should be called on a SEPARATE connection OUTSIDE the main write transaction
    to avoid holding the transaction open during expensive trigram scans.

    实现原理：
    1. retain 流程里，fact 先被抽取出来并生成占位 `unit_id`，但这时还没真正写入
       `memory_units`。
    2. 实体解析的目标，是把 fact 中出现的实体名称解析成系统里的 canonical entity id，
       并建立“哪个 unit 对应哪些 entity”的中间映射。
    3. 这个阶段只做“读”和“解析”，不做最终 entity link 写入，所以可以放在主写事务外，
       避免长事务被 trigram / 相似匹配查询拖住。
    4. 本函数本身只是一个 orchestration wrapper：
       - 先把 `ProcessedFact` 整理成轻量输入
       - 再调用 `link_utils.resolve_entities_only(...)`
       - 返回后续 `build_entity_links()` 所需的三个中间结果
    5. 这里返回的 `unit_ids` 仍然可能是占位 ID；真正写入数据库后，上游会把这些映射
       remap 到真实的 `memory_units.id`。

    Args:
        entity_resolver: EntityResolver instance
        conn: Database connection (separate from the main write transaction)
        bank_id: Bank identifier
        unit_ids: Placeholder unit IDs (used only for grouping)
        facts: List of ProcessedFact objects
        log_buffer: Optional buffer for detailed logging
        user_entities_per_content: Dict mapping content_index to user-provided entities
        entity_labels: Optional entity label taxonomy

    Returns:
        Tuple of (resolved_entity_ids, entity_to_unit, unit_to_entity_ids)
        to pass to build_entity_links().
    """
    # 没有 unit 或没有 facts 时，不需要做任何实体解析，直接返回空结构。
    if not unit_ids or not facts:
        return [], [], {}

    # `unit_ids` 和 `facts` 必须一一对应。
    # 每条 fact 后面解析出的实体，都要挂到同索引位置的 unit 上。
    if len(unit_ids) != len(facts):
        raise ValueError(f"Mismatch between unit_ids ({len(unit_ids)}) and facts ({len(facts)})")

    # 先把 retain 内部的 ProcessedFact 转成 resolver 需要的三类基础输入：
    # 文本、时间、实体候选列表。
    fact_texts, fact_dates, entities_per_fact = _prepare_facts_for_entity_processing(facts, user_entities_per_content)

    # 真正的实体解析在 link_utils.resolve_entities_only() 里执行。
    # 这里传入:
    # - unit_ids: 当前 facts 对应的占位/真实 unit 标识
    # - fact_texts: 用于辅助实体匹配/消歧
    # - fact_dates: 供 resolver 在需要时利用时间上下文
    # - entities_per_fact: 每条 fact 的候选实体名列表
    # 返回值包括:
    # - resolved_entity_ids: 本批次涉及到的 canonical entity id
    # - entity_to_unit: entity 与 unit 的对应关系
    # - unit_to_entity_ids: 每个 unit 关联到哪些 entity
    return await link_utils.resolve_entities_only(
        entity_resolver,
        conn,
        bank_id,
        unit_ids,
        fact_texts,
        "",  # context 预留给更复杂的解析策略；当前实现未实际使用。
        fact_dates,
        entities_per_fact,
        log_buffer,
        entity_labels=entity_labels,
    )


async def build_entity_links(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    resolved_entity_ids: list[str],
    entity_to_unit: list[tuple],
    unit_to_entity_ids: dict[str, list[str]],
    log_buffer: list[str] = None,
    skip_unit_entities_insert: bool = False,
) -> list[EntityLink]:
    """
    Build entity links for UI graph visualization.

    Queries unit_entities to find shared entities between new and existing units,
    then generates EntityLink objects. When called from Phase 3 (post-transaction),
    set skip_unit_entities_insert=True since unit_entities were already inserted
    in Phase 2.

    Args:
        entity_resolver: EntityResolver instance
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: Actual unit IDs (must already be inserted in the DB)
        resolved_entity_ids: From resolve_entities()
        entity_to_unit: From resolve_entities()
        unit_to_entity_ids: From resolve_entities()
        log_buffer: Optional buffer for detailed logging
        skip_unit_entities_insert: Skip unit_entities INSERT (already done in Phase 2)

    Returns:
        List of EntityLink objects for batch insertion
    """
    return await link_utils.build_entity_links_from_resolved(
        entity_resolver,
        conn,
        bank_id,
        unit_ids,
        resolved_entity_ids,
        entity_to_unit,
        unit_to_entity_ids,
        log_buffer,
        skip_unit_entities_insert=skip_unit_entities_insert,
    )


async def insert_entity_links_batch(conn, entity_links: list[EntityLink], bank_id: str) -> None:
    """
    Insert entity links in batch.

    Args:
        conn: Database connection
        entity_links: List of EntityLink objects
        bank_id: Bank identifier (stored directly on memory_links for fast filtering)
    """
    if not entity_links:
        return

    await link_utils.insert_entity_links_batch(conn, entity_links, bank_id)
