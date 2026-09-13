"""
Top-level RAG pipeline: combines retrieval + generation into one function.

This mirrors rag_query() from the notebook's Section 2.7 — it is the ONLY
function main.py's /query endpoint calls. main.py doesn't need to know
anything about ChromaDB, embedding models, or prompt templates.
"""

from app.rag.generator import generator
from app.rag.retriever import retriever


def rag_query(question: str, top_k: int = 3) -> dict:
    """Retrieve relevant chunks, generate a grounded answer, and package both
    together with a simple confidence score.

    Returns a plain dict (not a Pydantic model) — main.py is responsible for
    validating/shaping this into the QueryResponse schema. Keeping this
    function framework-agnostic makes it easy to reuse outside FastAPI too
    (e.g. a CLI script or a test) if needed later.
    """
    retrieved = retriever.retrieve(question, top_k=top_k)

    if not retrieved:
        return {
            "answer": "I don't have documentation on that.",
            "sources": [],
            "retrieved_chunks": [],
            "confidence": 0.0,
        }

    answer = generator.generate(question, retrieved)

    sources = [
        f"{c['source']} — {c['section']} (page {c['pages']})"
        for c in retrieved
    ]

    # distance is a cosine distance (lower = more similar); convert to a
    # simple 0-1 "confidence" so API clients don't need to know anything
    # about embeddings or distance metrics.
    top_distance = retrieved[0]["distance"]
    confidence = max(0.0, 1.0 - top_distance)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {
                "source": c["source"],
                "section": c["section"],
                "pages": c["pages"],
                "distance": c["distance"],
            }
            for c in retrieved
        ],
        "confidence": round(confidence, 3),
    }
