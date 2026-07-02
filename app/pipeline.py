import logging
from typing import Any, Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field

from app.agents.graph_builder import build_workflow
from app.config import settings
from app.graph_store import CognitiveGraphStore
from app.vector_store import CognitiveVectorStore

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    name: str = Field(description="Normalized entity name.")
    type: str = Field(description="Entity category, such as Person, Technology, Organization, or Concept.")
    description: str = Field(description="Short description of the entity.")


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Source entity name.")
    target: str = Field(description="Target entity name.")
    relationship: str = Field(description="Relationship type, such as DEVELOPS, USES, LEADS, INTEGRATES_WITH.")
    description: str = Field(description="Short relationship description.")


class KnowledgeGraphExtraction(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


class CognitiveGraphRAGPipeline:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        self.graph_store = CognitiveGraphStore(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
        )

        self.vector_store = CognitiveVectorStore(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.QDRANT_COLLECTION,
            openai_api_key=settings.OPENAI_API_KEY,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
        )

        self.workflow = build_workflow(
            openai_client=self.openai_client,
            model_name=settings.OPENAI_MODEL_NAME,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
        )

    def close(self) -> None:
        self.graph_store.close()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1200,
        overlap: int = 150,
    ) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    def extract_graph_from_chunk(self, chunk: str) -> KnowledgeGraphExtraction:
        prompt = f"""
Extract a compact knowledge graph from this text.

Return:
- important entities
- direct relationships between those entities

Text:
{chunk}
"""

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=settings.OPENAI_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract entities and relationships for a Neo4j knowledge graph.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=KnowledgeGraphExtraction,
            )

            return completion.choices[0].message.parsed

        except Exception as error:
            logger.exception("Graph extraction failed: %s", error)
            return KnowledgeGraphExtraction()

    def index_document(
        self,
        title: str,
        content: str,
        source: str = "manual",
    ) -> Dict[str, Any]:
        chunks = self.chunk_text(content)

        for index, chunk in enumerate(chunks):
            graph_data = self.extract_graph_from_chunk(chunk)

            for entity in graph_data.entities:
                self.graph_store.add_entity(
                    name=entity.name,
                    entity_type=entity.type,
                    description=entity.description,
                )

            for relationship in graph_data.relationships:
                self.graph_store.add_relationship(
                    source=relationship.source,
                    target=relationship.target,
                    relationship_type=relationship.relationship,
                    description=relationship.description,
                )

        self.vector_store.upsert_chunks(
            texts=chunks,
            metadatas=[
                {
                    "title": title,
                    "source": source,
                    "chunk_id": index,
                }
                for index in range(len(chunks))
            ],
        )

        return {
            "status": "success",
            "title": title,
            "source": source,
            "chunks_processed": len(chunks),
        }

    def query(self, query_text: str) -> Dict[str, Any]:
        initial_state = {
            "query": query_text,
            "refined_query": None,
            "extracted_entities": [],
            "vector_context": [],
            "graph_context": "",
            "combined_context": None,
            "response": None,
            "route": None,
            "retrieval_score": None,
            "hallucination_score": None,
            "answer_score": None,
            "loop_count": 0,
            "max_loops": 3,
        }

        final_state = self.workflow.invoke(initial_state)

        return {
            "query": query_text,
            "refined_query": final_state.get("refined_query"),
            "extracted_entities": final_state.get("extracted_entities"),
            "route": final_state.get("route"),
            "retrieval_score": final_state.get("retrieval_score"),
            "hallucination_score": final_state.get("hallucination_score"),
            "answer_score": final_state.get("answer_score"),
            "response": final_state.get("response"),
        }
