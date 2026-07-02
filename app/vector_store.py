import hashlib
from typing import Any, Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


class CognitiveVectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        openai_api_key: str,
        embedding_model: str,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=1536,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    def get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[text.replace("\n", " ")],
        )
        return response.data[0].embedding

    def upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        points = []

        for index, (text, metadata) in enumerate(zip(texts, metadatas)):
            vector = self.get_embedding(text)
            key = f"{metadata.get('source', 'manual')}-{index}-{text}"
            point_id = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) & 0xFFFFFFFFFFFFFFFF

            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={**metadata, "text": text},
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def similarity_search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        query_vector = self.get_embedding(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
        )

        hits = []

        for hit in results:
            payload = hit.payload or {}
            hits.append(
                {
                    "text": payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                }
            )

        return hits
