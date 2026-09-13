"""
Tests for the /health and /query endpoints.

The RAG components (embedding model, ChromaDB, Ollama) are mocked here so
these tests run fast and don't require a real vector store or a running
Ollama server — CI or a fresh clone can run `pytest` immediately. This only
tests the API layer (routing, validation, response shape, error handling);
it does NOT test retrieval/generation quality — that's what the notebook's
Section 2.5 Evaluation covers.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.generator import generator
from app.rag.retriever import retriever


@pytest.fixture
def client(monkeypatch):
    """A TestClient with retriever/generator mocked as 'ready', and
    retrieve()/generate() returning fixed, realistic fake data.
    """
    # Skip the real load() calls (would hit HuggingFace/Ollama over the network)
    monkeypatch.setattr(retriever, "load", lambda: setattr(retriever, "ready", True))
    monkeypatch.setattr(generator, "load", lambda: setattr(generator, "ready", True))

    with TestClient(app) as test_client:
        # Now stub the actual retrieval/generation behavior
        monkeypatch.setattr(
            retriever,
            "retrieve",
            lambda question, top_k=3: [{
                "text": "Resetting Your Password\n1.1 Click Forgot Password...",
                "source": "password_and_login.pdf",
                "category": "account",
                "section": "Resetting Your Password",
                "pages": "1",
                "distance": 0.185,
            }],
        )
        monkeypatch.setattr(
            generator,
            "generate",
            lambda question, chunks: "Click Forgot Password on the login screen.",
        )
        yield test_client


def test_health_reports_ready(client):
    """/health should report rag_ready=True once retriever+generator loaded."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["rag_ready"] is True
    assert body["llm_model"] == "llama3.2"


def test_query_happy_path(client):
    """A valid question returns a 200 with the expected QueryResponse shape."""
    response = client.post("/query", json={"question": "How do I reset my password?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Click Forgot Password on the login screen."
    assert body["sources"] == ["password_and_login.pdf — Resetting Your Password (page 1)"]
    assert len(body["retrieved_chunks"]) == 1
    assert body["retrieved_chunks"][0]["source"] == "password_and_login.pdf"
    # confidence = 1 - distance = 1 - 0.185
    assert body["confidence"] == pytest.approx(0.815)


def test_query_empty_question_returns_422(client):
    """An empty question should fail Pydantic validation (min_length=1)."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_top_k_out_of_range_returns_422(client):
    """top_k must be between 1 and 10 per the QueryRequest schema."""
    response = client.post("/query", json={"question": "test", "top_k": 50})
    assert response.status_code == 422


def test_query_missing_question_returns_422(client):
    """The `question` field is required."""
    response = client.post("/query", json={})
    assert response.status_code == 422
