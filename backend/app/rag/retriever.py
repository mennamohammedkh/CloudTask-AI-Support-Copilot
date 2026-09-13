"""
Retrieval component of the RAG pipeline.

Wraps the embedding model + persisted ChromaDB collection from the notebook
(Section 2.3) behind a small class, so `main.py` can create ONE instance at
application startup and reuse it for every request instead of reloading the
model or reopening the vector store per call.
"""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self._embedding_model: SentenceTransformer | None = None
        self._collection = None
        self.ready = False
        self.error: str | None = None

    def load(self) -> None:
        """Load the embedding model and open the persisted ChromaDB collection.

        Called once at app startup (see main.py's lifespan). Any failure here
        is caught and stored in `self.error` rather than raised, so the app
        can still start and report `rag_ready: false` via /health instead of
        crashing outright.
        """
        try:
            logger.info("Loading embedding model: %s", settings.embedding_model)
            self._embedding_model = SentenceTransformer(settings.embedding_model)

            logger.info("Opening ChromaDB at: %s", settings.vector_store_path)
            client = chromadb.PersistentClient(path=settings.vector_store_path)
            self._collection = client.get_collection(name=settings.vector_store_collection)

            count = self._collection.count()
            if count == 0:
                raise RuntimeError(
                    f"Collection '{settings.vector_store_collection}' exists but is "
                    "empty — did you run the notebook's Section 2.3 and 2.6 Export?"
                )

            logger.info("Retriever ready: %d chunks loaded.", count)
            self.ready = True

        except Exception as exc:
            # Common causes: vector_store_path doesn't exist yet, or the
            # collection name doesn't match what the notebook created.
            logger.exception("Failed to load retriever")
            self.error = str(exc)
            self.ready = False

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most relevant chunks for `question`.

        Mirrors the retrieve() function from the notebook's Section 2.4 —
        same normalize_embeddings=True setting, same metadata shape.
        """
        if not self.ready:
            raise RuntimeError(
                "Retriever is not ready. Check /health for the error, and "
                "make sure the vector store was copied into backend/data/."
            )

        query_embedding = self._embedding_model.encode(
            [question], normalize_embeddings=True
        ).tolist()

        results = self._collection.query(query_embeddings=query_embedding, n_results=top_k)

        if not results["documents"] or not results["documents"][0]:
            return []

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta["source"],
                "category": meta["category"],
                "section": meta["section"],
                "pages": meta["pages"],
                "distance": dist,
            })
        return retrieved


# A single shared instance. main.py calls `retriever.load()` once at startup
# and every request handler reuses this same object.
retriever = Retriever()
