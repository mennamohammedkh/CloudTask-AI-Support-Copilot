"""
CloudTask AI Support Copilot — Streamlit frontend.

Run locally with:
    streamlit run app.py
"""

import streamlit as st

from api_client import BackendError, check_health, get_backend_url, query_backend

st.set_page_config(page_title="CloudTask Support Copilot", page_icon="🤖")

st.title("🤖 CloudTask Support Copilot")
st.caption(f"Backend: {get_backend_url()}")

# --- Backend status banner ---
# Checked once per session load (not on every message) so the UI doesn't
# add an extra request on every question just to show a status dot.
if "backend_checked" not in st.session_state:
    try:
        health = check_health()
        st.session_state.backend_checked = True
        st.session_state.backend_ready = health.get("rag_ready", False)
    except BackendError as e:
        st.session_state.backend_checked = True
        st.session_state.backend_ready = False
        st.session_state.backend_error = str(e)

if not st.session_state.get("backend_ready"):
    st.warning(
        st.session_state.get(
            "backend_error",
            "The backend reports the RAG pipeline isn't ready yet "
            "(vector store or Ollama may still be loading).",
        )
    )

# --- Chat history (kept in session_state so it survives reruns) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
        if message.get("confidence") is not None:
            st.caption(f"Confidence: {message['confidence']:.0%}")

# --- New question input ---
question = st.chat_input("Ask a question about CloudTask...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_backend(question, top_k=3)
                answer = result["answer"]
                sources = result.get("sources", [])
                confidence = result.get("confidence")

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
                if confidence is not None:
                    st.caption(f"Confidence: {confidence:.0%}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })

            except BackendError as e:
                # Friendly error message, not a raw traceback, per the
                # assignment's frontend requirements.
                error_message = f"⚠️ {e}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })
