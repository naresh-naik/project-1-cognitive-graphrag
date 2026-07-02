"""Unit tests for CognitiveGraphStore and CognitiveVectorStore with mocked drivers."""

from unittest.mock import MagicMock, patch

from app.graph_store import CognitiveGraphStore
from app.vector_store import CognitiveVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_neo4j_driver(mock_driver_cls):
    """Wire a mock Neo4j driver and session into ``GraphDatabase.driver``."""
    mock_driver = MagicMock()
    mock_driver_cls.return_value = mock_driver

    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None
    mock_driver.session.return_value = mock_session

    return mock_driver, mock_session


def _mock_qdrant_client(mock_qdrant_cls, existing_collections=None):
    """Wire a mock Qdrant client into ``QdrantClient``."""
    mock_client = MagicMock()
    mock_qdrant_cls.return_value = mock_client

    collections = []
    for name in existing_collections or []:
        collection = MagicMock()
        collection.name = name
        collections.append(collection)
    mock_client.get_collections.return_value.collections = collections

    return mock_client


# ---------------------------------------------------------------------------
# CognitiveGraphStore tests
# ---------------------------------------------------------------------------


@patch("app.graph_store.GraphDatabase.driver")
def test_graph_store_init_connects_and_creates_indexes(mock_driver_cls):
    """__init__ should create the driver, verify connectivity, and build indexes."""
    mock_driver, mock_session = _mock_neo4j_driver(mock_driver_cls)

    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")

    mock_driver_cls.assert_called_once_with(
        "bolt://localhost:7687", auth=("neo4j", "password")
    )
    mock_driver.verify_connectivity.assert_called_once()
    assert mock_session.run.call_count == 2
    store.close()


@patch("app.graph_store.GraphDatabase.driver")
def test_add_entity_runs_merge_query(mock_driver_cls):
    """add_entity should strip input and execute the merge query."""
    _, mock_session = _mock_neo4j_driver(mock_driver_cls)
    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")
    mock_session.reset_mock()

    store.add_entity("  GraphRAG  ", "  Concept  ", "  retrieval technique  ")

    mock_session.run.assert_called_once()
    args, kwargs = mock_session.run.call_args
    assert "MERGE (e:Entity" in args[0]
    assert kwargs["name"] == "GraphRAG"
    assert kwargs["type"] == "Concept"
    assert kwargs["description"] == "retrieval technique"
    store.close()


@patch("app.graph_store.GraphDatabase.driver")
def test_add_relationship_normalizes_relationship_type(mock_driver_cls):
    """add_relationship should sanitize relationship types for Cypher."""
    _, mock_session = _mock_neo4j_driver(mock_driver_cls)
    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")
    mock_session.reset_mock()

    store.add_relationship("Source", "Target", "relates to!", "description")

    query = mock_session.run.call_args[0][0]
    assert "-[r:RELATES_TO]->" in query
    store.close()


@patch("app.graph_store.GraphDatabase.driver")
def test_fuzzy_match_entities_filters_by_score(mock_driver_cls):
    """fuzzy_match_entities should return only records above the score threshold."""
    _, mock_session = _mock_neo4j_driver(mock_driver_cls)
    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")
    mock_session.reset_mock()

    mock_session.run.return_value = [
        {"name": "GraphRAG", "score": 0.9},
        {"name": "Irrelevant", "score": 0.1},
    ]

    matches = store.fuzzy_match_entities("graphrag", limit=3)

    assert matches == ["GraphRAG"]
    args, kwargs = mock_session.run.call_args
    assert kwargs["query_text"] == "graphrag~2"
    assert kwargs["limit"] == 3
    store.close()


@patch("app.graph_store.GraphDatabase.driver")
def test_get_entity_context_formats_output(mock_driver_cls):
    """get_entity_context should assemble entity and relationship snippets."""
    _, mock_session = _mock_neo4j_driver(mock_driver_cls)
    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")
    mock_session.reset_mock()

    mock_session.run.return_value = [
        {
            "source": "GraphRAG",
            "source_type": "Concept",
            "source_description": "A retrieval technique.",
            "relationship_type": "IMPROVES",
            "relationship_description": "enhances",
            "target": "Retrieval",
            "target_type": "Process",
        }
    ]

    context = store.get_entity_context(["GraphRAG"])

    assert "Entity: GraphRAG (Concept) - A retrieval technique." in context
    assert "Relationship: (GraphRAG) -[IMPROVES: enhances]-> (Retrieval)" in context
    store.close()


@patch("app.graph_store.GraphDatabase.driver")
def test_get_entity_context_empty_entities_returns_empty_string(mock_driver_cls):
    """get_entity_context should short-circuit when no entities are supplied."""
    _, mock_session = _mock_neo4j_driver(mock_driver_cls)
    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")
    mock_session.reset_mock()

    assert store.get_entity_context([]) == ""
    mock_session.run.assert_not_called()
    store.close()


@patch("app.graph_store.GraphDatabase.driver")
def test_graph_store_close(mock_driver_cls):
    """close should delegate to the underlying driver."""
    mock_driver, _ = _mock_neo4j_driver(mock_driver_cls)
    store = CognitiveGraphStore("bolt://localhost:7687", "neo4j", "password")

    store.close()

    mock_driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# CognitiveVectorStore tests
# ---------------------------------------------------------------------------


@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_vector_store_init_skips_existing_collection(mock_qdrant_cls, _mock_openai_cls):
    """__init__ should not recreate an existing collection."""
    mock_client = _mock_qdrant_client(mock_qdrant_cls, existing_collections=["test_collection"])

    CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    mock_client.get_collections.assert_called_once()
    mock_client.create_collection.assert_not_called()


@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_vector_store_init_creates_missing_collection(mock_qdrant_cls, _mock_openai_cls):
    """__init__ should create the collection when it does not yet exist."""
    mock_client = _mock_qdrant_client(mock_qdrant_cls, existing_collections=[])

    CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    assert call_kwargs["vectors_config"].size == 1536


@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_get_embedding_calls_openai_and_normalizes_input(mock_qdrant_cls, mock_openai_cls):
    """get_embedding should request embeddings and collapse newlines."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai_cls.return_value.embeddings.create.return_value = mock_response

    store = CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    vector = store.get_embedding("hello\nworld")

    mock_openai_cls.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["hello world"],
    )
    assert vector == [0.1, 0.2, 0.3]


@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_upsert_chunks_empty_input_is_noop(mock_qdrant_cls, _mock_openai_cls):
    """upsert_chunks with empty input should not call the Qdrant upsert API."""
    mock_client = _mock_qdrant_client(mock_qdrant_cls, existing_collections=["test_collection"])
    store = CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    store.upsert_chunks([], [])

    mock_client.upsert.assert_not_called()


@patch("app.vector_store.CognitiveVectorStore.get_embedding")
@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_upsert_chunks_builds_point_structs(mock_qdrant_cls, _mock_openai_cls, mock_embed):
    """upsert_chunks should embed texts and upsert Qdrant PointStruct objects."""
    mock_client = _mock_qdrant_client(mock_qdrant_cls, existing_collections=["test_collection"])
    mock_embed.return_value = [0.5] * 1536

    store = CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    store.upsert_chunks(
        texts=["GraphRAG combines graphs and RAG."],
        metadatas=[{"source": "docs", "page": 1}],
    )

    mock_client.upsert.assert_called_once()
    call_kwargs = mock_client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    assert len(call_kwargs["points"]) == 1

    point = call_kwargs["points"][0]
    assert point.vector == [0.5] * 1536
    assert point.payload["text"] == "GraphRAG combines graphs and RAG."
    assert point.payload["source"] == "docs"
    assert point.payload["page"] == 1


@patch("app.vector_store.CognitiveVectorStore.get_embedding")
@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_similarity_search_returns_payload_and_metadata(mock_qdrant_cls, _mock_openai_cls, mock_embed):
    """similarity_search should transform Qdrant hits into text/score/metadata dicts."""
    mock_client = _mock_qdrant_client(mock_qdrant_cls, existing_collections=["test_collection"])
    mock_embed.return_value = [0.1] * 1536

    hit = MagicMock()
    hit.payload = {"text": "GraphRAG improves retrieval.", "source": "docs"}
    hit.score = 0.87
    mock_client.search.return_value = [hit]

    store = CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    results = store.similarity_search("What is GraphRAG?", limit=2)

    mock_client.search.assert_called_once_with(
        collection_name="test_collection",
        query_vector=[0.1] * 1536,
        limit=2,
    )
    assert len(results) == 1
    assert results[0]["text"] == "GraphRAG improves retrieval."
    assert results[0]["score"] == 0.87
    assert results[0]["metadata"] == {"source": "docs"}


@patch("app.vector_store.CognitiveVectorStore.get_embedding")
@patch("app.vector_store.OpenAI")
@patch("app.vector_store.QdrantClient")
def test_similarity_search_handles_empty_payload(mock_qdrant_cls, _mock_openai_cls, mock_embed):
    """similarity_search should tolerate hits with no payload."""
    mock_client = _mock_qdrant_client(mock_qdrant_cls, existing_collections=["test_collection"])
    mock_embed.return_value = [0.2] * 1536

    hit = MagicMock()
    hit.payload = None
    hit.score = 0.5
    mock_client.search.return_value = [hit]

    store = CognitiveVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    results = store.similarity_search("query", limit=1)

    assert results == [{"text": "", "score": 0.5, "metadata": {}}]
