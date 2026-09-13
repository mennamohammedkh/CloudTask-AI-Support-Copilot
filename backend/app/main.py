"""
CloudTask AI Support Copilot — FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.rag.generator import generator
from app.rag.pipeline import rag_query
from app.rag.retriever import retriever
from app.schemas import HealthResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server starts — loads the embedding model + vector
    # store and checks Ollama, so /query never pays that cost per-request.
    logger.info("Starting up: loading retriever and generator...")
    retriever.load()
    generator.load()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="CloudTask AI Support Copilot",
    description="RAG-powered backend that answers CloudTask support questions "
                "using the official documentation.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the frontend (Streamlit/Gradio, running on a different port) to call
# this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness + readiness check.

    `rag_ready` is True only if BOTH the vector store/embedding model
    (retriever) AND Ollama (generator) loaded successfully at startup.
    """
    return HealthResponse(
        status="ok",
        rag_ready=retriever.ready and generator.ready,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a customer support question using the RAG pipeline.

    Validation of `question` (non-empty, max length) and `top_k` (1-10) is
    already handled by the QueryRequest schema before this function runs.
    """
    if not (retriever.ready and generator.ready):
        # 503 = "service temporarily unavailable", not a client error (400s).
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready. Check /health for details.",
        )

    try:
        result = rag_query(request.question, top_k=request.top_k)
    except Exception:
        # Never leak internal stack traces to the client — log the real
        # error server-side and return a generic message instead.
        logger.exception("rag_query failed for question: %r", request.question)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating an answer. Please try again.",
        )

    return QueryResponse(**result)
