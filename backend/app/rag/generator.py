"""
Generation component of the RAG pipeline.

Combines retrieved chunks + the user's question into a grounded prompt, then
calls the local Ollama LLM. Mirrors build_prompt() / call_llm() from the
notebook's Section 2.4, wrapped in a class so main.py can check Ollama's
availability once at startup (similar to how Retriever checks the vector
store).
"""

import logging

import ollama

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are a customer support copilot for CloudTask, a project \
management SaaS product. Answer the user's question using ONLY the context \
provided below, taken from CloudTask's official documentation.

Rules:
- If the answer is not contained in the context, say so explicitly ("I don't have documentation on that") instead of guessing.
- Do not invent policies, numbers, or steps that are not in the context.
- Keep the answer concise and direct.
- After the answer, list the sources you used.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Combine retrieved chunks + the question into the final LLM prompt.

    Identical logic to the notebook's build_prompt() (Section 2.4) — kept
    here rather than imported from the notebook so the backend has no
    runtime dependency on Jupyter/nbformat.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} — \"{chunk['section']}\" "
            f"(page {chunk['pages']})]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""{SYSTEM_INSTRUCTIONS}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class Generator:
    def __init__(self):
        self.ready = False
        self.error: str | None = None

    def load(self) -> None:
        """Verify Ollama is reachable and the configured model responds.

        Called once at app startup. Like Retriever.load(), failures are
        caught and stored rather than raised, so /health can report the
        problem instead of the app crashing on import.
        """
        try:
            logger.info("Checking Ollama at %s for model %s",
                        settings.ollama_host, settings.llm_model)
            # A minimal real call rather than just pinging the server, so we
            # also catch "server is up but model isn't pulled" errors.
            ollama.chat(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "Reply with: ok"}],
            )
            self.ready = True
            logger.info("Generator ready: Ollama model '%s' responded.", settings.llm_model)
        except Exception as exc:
            logger.exception("Failed to reach Ollama")
            self.error = str(exc)
            self.ready = False

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Build the grounded prompt and call the LLM. Returns the raw answer text."""
        if not self.ready:
            raise RuntimeError(
                "Generator is not ready. Check /health for the error — Ollama "
                "may not be running, or the model may not be pulled "
                f"(try: ollama pull {settings.llm_model})."
            )

        prompt = build_prompt(question, retrieved_chunks)
        response = ollama.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# Shared instance, loaded once at startup (see main.py).
generator = Generator()
