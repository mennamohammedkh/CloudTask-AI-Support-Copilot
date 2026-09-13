"""
Pydantic schemas for the CloudTask Support Copilot API.

These define the exact JSON shape clients (the frontend) send and receive.
FastAPI uses them to validate incoming requests automatically and to
document the API in /docs.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question in plain text.",
        examples=["How do I reset my password?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve from the vector store (1-10).",
    )


class RetrievedChunk(BaseModel):
    source: str = Field(..., description="Source PDF filename, e.g. 'refund_policy.pdf'.")
    section: str = Field(..., description="Section heading the chunk was taken from.")
    pages: str = Field(..., description="Page number(s) the chunk appears on, e.g. '1' or '1,2'.")
    distance: float = Field(..., description="Cosine distance to the query (lower = more similar).")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The LLM's grounded answer.")
    sources: list[str] = Field(
        default_factory=list,
        description="Human-readable citations, e.g. 'refund_policy.pdf — Refund Eligibility (page 1)'.",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="The raw retrieved chunks, for clients that want more detail than `sources`.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score derived from the top retrieved chunk's similarity.",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the service is reachable.")
    service: str = "CloudTask AI Support Copilot"
    rag_ready: bool = Field(
        ..., description="True only if the vector store and embedding model loaded successfully."
    )
    llm_model: str
    embedding_model: str
