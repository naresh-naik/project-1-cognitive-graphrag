"""FastAPI TestClient smoke tests for the Cognitive GraphRAG API."""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


class MockPipeline:
    """Deterministic stand-in for CognitiveGraphRAGPipeline."""

    def __init__(self):
        self.closed = False

    def index_document(self, title: str, content: str, source: str):
        return {
            "success": True,
            "title": title,
            "source": source,
            "chunks_indexed": 1,
        }

    def query(self, query: str):
        return {
            "response": f"Mock answer for: {query}",
            "sources": ["mock-source"],
        }

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    """Replace the real pipeline class so no live services are contacted."""
    monkeypatch.setattr(main_module, "CognitiveGraphRAGPipeline", MockPipeline)


@pytest.fixture
def client():
    """Yield a TestClient with a mocked pipeline lifespan."""
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_returns_200_and_status(client):
    """GET /health should report the API is healthy."""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "project" in payload


def test_index_returns_201_and_success(client):
    """POST /index should return a successful indexing result."""
    payload = {
        "title": "Quantum Research Lab",
        "content": "Dr. Aris Thorne leads the Quantum Materials Division.",
        "source": "manual",
    }
    response = client.post("/index", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["title"] == payload["title"]
    assert data["source"] == payload["source"]


def test_query_returns_200_and_answer(client):
    """POST /query should return a deterministic mocked answer."""
    payload = {"query": "Who leads the Quantum Materials Division?"}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] == f"Mock answer for: {payload['query']}"
    assert "sources" in data


def test_index_without_pipeline_ready_returns_503(monkeypatch, client):
    """POST /index should return 503 when the global pipeline is None."""
    monkeypatch.setattr(main_module, "pipeline", None)

    response = client.post(
        "/index",
        json={
            "title": "Test",
            "content": "Test content.",
            "source": "manual",
        },
    )

    assert response.status_code == 503
    assert "Pipeline is not ready" in response.json()["detail"]


def test_query_without_pipeline_ready_returns_503(monkeypatch, client):
    """POST /query should return 503 when the global pipeline is None."""
    monkeypatch.setattr(main_module, "pipeline", None)

    response = client.post("/query", json={"query": "What is GraphRAG?"})

    assert response.status_code == 503
    assert "Pipeline is not ready" in response.json()["detail"]
