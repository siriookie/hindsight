"""
Embedding processing for retain pipeline.

Handles augmenting fact texts with temporal information and generating embeddings.
"""

import logging

from . import embedding_utils
from .types import ExtractedFact

logger = logging.getLogger(__name__)


def augment_texts_with_dates(facts: list[ExtractedFact], format_date_fn) -> list[str]:
    """
    Augment fact texts with readable dates for better temporal matching.

    This allows queries like "camping in June" to match facts that happened in June.

    Args:
        facts: List of ExtractedFact objects
        format_date_fn: Function to format datetime to readable string

    Returns:
        List of augmented text strings (same length as facts)
    """
    # 最终返回的增强文本列表，与输入 facts 一一对应。
    augmented_texts = []
    for fact in facts:
        # 优先使用事实真正发生的时间 occurred_start；
        # 如果没有发生时间，再退回到“被提到/被记录”的时间 mentioned_at。
        fact_date = fact.occurred_start or fact.mentioned_at

        # 这里增强的是“送去做 embedding 的文本”，不是数据库里最终存储的原始 fact_text。
        # 目的是让向量检索更容易命中时间表达和实体名，而不污染原始内容。
        if fact_date is not None:
            # 把 datetime 转成更适合语义检索的可读日期字符串。
            readable_date = format_date_fn(fact_date)
            if fact.occurred_end and fact.occurred_end != fact.occurred_start:
                # 如果这是一个有持续时间的事件，就把开始和结束时间都拼进去。
                readable_end = format_date_fn(fact.occurred_end)
                augmented_text = f"{fact.fact_text} (happened from {readable_date} to {readable_end})"
            else:
                # 如果只有一个代表性时间点，就只拼一个时间表达。
                augmented_text = f"{fact.fact_text} (happened in {readable_date})"
        else:
            # 完全没有可用时间时，embedding 文本就保持原始 fact_text。
            augmented_text = fact.fact_text
        if fact.entities:
            # 把实体名也附加到 embedding 文本里，帮助检索时利用实体线索。
            # 这里的 entities 也可能包含 key:value 形式的 labels。
            augmented_text = f"{augmented_text} [{', '.join(fact.entities)}]"

        # 把当前 fact 对应的增强文本加入结果列表。
        augmented_texts.append(augmented_text)

    # 返回与输入 facts 等长的增强文本列表，供后续批量生成 embedding。
    return augmented_texts


async def generate_embeddings_batch(embeddings_model, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        embeddings_model: Embeddings model instance
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (same length as texts)
    """
    if not texts:
        return []

    embeddings = await embedding_utils.generate_embeddings_batch(embeddings_model, texts)

    return embeddings
