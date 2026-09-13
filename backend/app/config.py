"""
Central configuration for the CloudTask Support Copilot backend.

Every other module (main.py, rag/*) reads settings through the `settings`
object defined here instead of hard-coding values or reading os.environ
directly. This is the single place that knows about the .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Vector store ---
    vector_store_path: str = "./data/vector_store"
    vector_store_collection: str = "cloudtask_kb"

    # --- Embedding model (must match what the notebook used to build the
    # vector store, or query vectors won't align with stored chunk vectors) ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- LLM (Ollama) ---
    llm_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    # --- Retrieval defaults ---
    default_top_k: int = 3

    # --- CORS ---
    # Comma-separated in .env, e.g. "http://localhost:8501,http://localhost:7860"
    allowed_origins: str = "http://localhost:8501,http://localhost:7860"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parsed CORS origins, ready for FastAPI's CORSMiddleware."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


# A single shared instance, imported everywhere else as:
#   from app.config import settings
settings = Settings()
