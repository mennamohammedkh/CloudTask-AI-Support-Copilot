"""
Thin wrapper around the CloudTask Support Copilot backend API.

Keeping all HTTP/error-handling logic here means app.py only deals with
Streamlit UI code, and the backend URL is never hard-coded anywhere.
"""

import os

import requests

# Read from environment, with a sensible local-dev default. Never hard-code
# this elsewhere in the app — always go through get_backend_url()/query_backend().
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

REQUEST_TIMEOUT_SECONDS = 60  # LLM generation can take a while on CPU


class BackendError(Exception):
    """Raised for any problem talking to the backend, with a message that's
    already safe and friendly to show directly in the UI."""
    pass


def get_backend_url() -> str:
    return API_BASE_URL


def check_health() -> dict:
    """Call GET /health. Raises BackendError if the backend is unreachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise BackendError(
            f"Can't reach the backend at {API_BASE_URL}. "
            "Is it running? (uvicorn app.main:app --reload)"
        )
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Backend health check failed: {e}")


def query_backend(question: str, top_k: int = 3) -> dict:
    """Call POST /query and return the parsed JSON response.

    Raises BackendError with a user-friendly message for every failure mode:
    backend not running, backend not ready (RAG not loaded), validation
    error, or an unexpected server error.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        raise BackendError(
            f"Can't reach the backend at {API_BASE_URL}. "
            "Is it running? (uvicorn app.main:app --reload)"
        )
    except requests.exceptions.Timeout:
        raise BackendError(
            "The backend took too long to respond. The LLM may be slow on "
            "this machine — try again, or reduce top_k."
        )

    if response.status_code == 200:
        return response.json()

    if response.status_code == 422:
        raise BackendError("That question wasn't valid — please rephrase and try again.")

    if response.status_code == 503:
        raise BackendError(
            "The backend is still starting up (or the vector store / Ollama "
            "isn't ready). Check the backend's /health endpoint."
        )

    # Any other status: don't leak raw backend internals to the UI.
    raise BackendError(f"Backend returned an unexpected error (status {response.status_code}).")
