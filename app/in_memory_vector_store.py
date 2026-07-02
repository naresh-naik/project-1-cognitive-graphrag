import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


def _dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _magnitude(vector: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = _magnitude(a) * _magnitude(b)
    if denom == 0:
        return 0.0
    return _dot_product(a, b) / denom


class InMemoryCognitiveVectorStore:
    """In-memory fallback for CognitiveVectorStore when Qdrant is unavailable."""

    def __init__(
        self,
        *_args,
        openai_api_key: str,
        embedding_model: str,
        embedding_client: Optional[Any] = None,
        **_kwargs,
    ):
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client
        self.points: List[Dict[str, Any]] = []
        logger.info("Using in-memory vector store.")

    def _ensure_collection(self) -> None:
        pass

    def get_embedding(self, text: str) -> List[float]:
        if self.embedding_client is not None:
            return self.embedding_client.embed([text.replace("\n", " ")])[0]

        response = OpenAI(api_key=self.openai_api_key).embeddings.create(
            model=self.embedding_model,
            input=[text.replace("\n", " ")],
        )
        return response.data[0].embedding

    def upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for index, (text, metadata) in enumerate(zip(texts, metadatas)):
            vector = self.get_embedding(text)
            key = f"{metadata.get('source', 'manual')}-{index}-{text}"
            point_id = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) & 0xFFFFFFFFFFFFFFFF

            # Remove any existing point with the same id to mimic upsert
            self.points = [p for p in self.points if p["id"] != point_id]
            self.points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {**metadata, "text": text},
                }
            )

    def similarity_search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        query_vector = self.get_embedding(query)

        scored = []
        for point in self.points:
            score = _cosine_similarity(query_vector, point["vector"])
            scored.append((score, point))

        scored.sort(key=lambda item: item[0], reverse=True)

        hits = []
        for score, point in scored[:limit]:
            payload = point["payload"]
            hits.append(
                {
                    "text": payload.get("text", ""),
                    "score": score,
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                }
            )

        return hits
