"""
Streamlit front-end for the local RAG pipeline.

Lets the user upload any PDF, asks questions about it, and streams
the answer token-by-token from a fully local LLM — no cloud required.

Pipeline recap
--------------
PDF upload → PdfReader (text extraction) → LocalEmbedding (indexing)
→ AiModel.ask_a_question_from_pdf_stream (retrieval + generation)
→ st.write_stream (live token rendering in the browser)
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from local_llm import AiModel
from local_embedding import LocalEmbedding
from pdf_reader import PdfReader


# ------------------------------------------------------------------
# Page configuration  (must be the very first Streamlit call)
# ------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
# Streamlit reruns the whole script on every user interaction.
# Session state is the only thing that persists across those reruns
# within a single browser session.

for key, default in [
    ("pdf_name", None),         # filename — used to detect when the user switches PDFs
    ("tmp_pdf_path", None),     # path to the temp file PdfReader opens
    ("local_embedding", None),  # built index — avoids re-indexing on follow-up questions
    ("chat_history", []),       # list of {"role": ..., "content": ...} dicts
    ("paragraph_count", 0),     # displayed in the header once a doc is indexed
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ai_model() -> AiModel:
    '''
        Loads AiModel exactly once for the lifetime of the Streamlit server process.
        @st.cache_resource returns the same instance on every subsequent call,
        so the LLM is never loaded more than once regardless of reruns.
    '''
    load_dotenv()    # make HF_TOKEN available before AiModel.__init__ reads it
    return AiModel()


def save_uploaded_pdf(uploaded_file) -> str:
    '''
        Writes the uploaded PDF bytes to a NamedTemporaryFile on disk
        and returns the absolute path to that file.

        delete=False is required because PdfReader must open the file
        by path after this function returns — it cannot accept raw bytes.
    '''
    suffix = f"_{uploaded_file.name}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# ------------------------------------------------------------------
# Sidebar — model status + document upload
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📄 RAG Assistant")
    st.markdown("Ask questions about any PDF — fully local, no cloud required.")
    st.divider()

    # Model loading
    # show_spinner=False on the decorator lets us display our own richer status widget
    st.markdown("#### Model")
    with st.status("Loading LLM — this may take a minute on first run…", expanded=True) as model_status:
        ai_model = load_ai_model()
        model_status.update(label="LLM ready", state="complete", expanded=False)

    st.divider()

    # Document upload
    st.markdown("#### Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Drag and drop a PDF here, or click Browse files.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # When the user uploads a different PDF, tear down the previous state
        if uploaded_file.name != st.session_state.pdf_name:
            if st.session_state.tmp_pdf_path and os.path.exists(st.session_state.tmp_pdf_path):
                os.unlink(st.session_state.tmp_pdf_path)  # free the old temp file

            st.session_state.pdf_name = uploaded_file.name
            st.session_state.tmp_pdf_path = save_uploaded_pdf(uploaded_file)
            st.session_state.local_embedding = None   # force re-indexing for new doc
            st.session_state.chat_history = []        # clear history for new doc
            st.session_state.paragraph_count = 0

        # Build the embedding index the first time this PDF is uploaded
        if st.session_state.local_embedding is None:
            with st.status("Processing document…", expanded=True) as doc_status:
                st.write("Extracting text from PDF…")
                pdf_reader = PdfReader(st.session_state.tmp_pdf_path)
                paragraphs = pdf_reader.get_paragraphs()
                st.session_state.paragraph_count = len(paragraphs)

                st.write(f"Building embedding index for {len(paragraphs)} paragraphs…")
                embedding = LocalEmbedding()
                embedding.build_index(paragraphs)
                st.session_state.local_embedding = embedding

                doc_status.update(
                    label=f"Document ready — {len(paragraphs)} paragraphs indexed",
                    state="complete",
                    expanded=False,
                )
        else:
            # Subsequent reruns (e.g. user typed a question): show a simple badge
            st.success(
                f"{st.session_state.pdf_name}\n\n"
                f"{st.session_state.paragraph_count} paragraphs indexed"
            )
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    st.caption("Powered by **Qwen2.5-3B-Instruct** + **MiniLM-L6-v2** · runs entirely on your machine")


# ------------------------------------------------------------------
# Main content area — chat interface
# ------------------------------------------------------------------

st.markdown("# Ask Your Document")

if st.session_state.local_embedding is None:
    st.markdown("> **Upload a PDF in the sidebar** to begin asking questions.")
else:
    st.markdown(
        f"Chatting about **{st.session_state.pdf_name}** · "
        f"{st.session_state.paragraph_count} paragraphs indexed"
    )

st.divider()

# Replay the full conversation on every rerun so chat history stays visible
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input — pinned to the bottom of the page by Streamlit natively.
# Passing disabled=True prevents questions before a PDF is indexed.
prompt = st.chat_input(
    placeholder="Ask a question about your document…",
    disabled=(st.session_state.local_embedding is None),
)

if prompt:
    # Show the user's message immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant response token-by-token.
    # st.write_stream consumes the generator and renders each chunk as it arrives,
    # then returns the fully concatenated string when generation is complete.
    with st.chat_message("assistant"):
        response_stream = ai_model.ask_a_question_from_pdf_stream(
            pdf_path=st.session_state.tmp_pdf_path,
            prompt=prompt,
            local_embedding=st.session_state.local_embedding,
        )
        full_response = st.write_stream(response_stream)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})


###############
# Run
###############

# .\rag_env\Scripts\python.exe -m streamlit run main.py
# streamlit run main.py
