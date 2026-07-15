import logging
from typing import List

from app.config import get_settings
from app.dependencies import get_mistral_client

logger = logging.getLogger(__name__)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts in batches to avoid oversized Mistral requests
    (senior pattern: never embed a 500-page PDF in one API call).
    """
    if not texts:
        return []

    settings = get_settings()
    client = get_mistral_client()
    all_embeddings: List[List[float]] = []

    batch_size = settings.embed_batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.info("Embedding batch %s-%s / %s", start, start + len(batch), len(texts))
        response = client.embeddings.create(
            model=settings.embedding_model,
            inputs=batch,
        )
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings
