import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Unified embedding client supporting OpenAI or local sentence-transformers."""

    def __init__(
        self,
        provider: str,
        openai_api_key: str,
        openai_model: str,
        sentence_transformer_model: str,
    ):
        self.provider = provider.lower()
        self.openai_model = openai_model
        self.sentence_transformer_model = sentence_transformer_model

        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=openai_api_key)
        elif self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer

                self.local_model = SentenceTransformer(sentence_transformer_model)
                logger.info("Loaded local embedding model: %s", sentence_transformer_model)
            except Exception as error:
                logger.exception("Failed to load local embedding model: %s", error)
                raise
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            response = self.openai_client.embeddings.create(
                model=self.openai_model,
                input=[text.replace("\n", " ") for text in texts],
            )
            return [item.embedding for item in response.data]

        # local sentence-transformers
        import numpy as np

        embeddings = self.local_model.encode(texts, convert_to_numpy=True)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings
