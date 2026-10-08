# RAG AI System — Ask Your Documents, Privately

> A fully local Retrieval-Augmented Generation (RAG) pipeline with a Streamlit chat UI — upload any PDF, ask questions in plain English, and get grounded, streamed answers. No cloud inference APIs. No data leaving your machine. Runs on a laptop GPU using a 4-bit quantized LLM.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct_4bit-a78bfa)](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-bnb-4bit)

---

## What Is This?

RAG AI System is a fully local Retrieval-Augmented Generation pipeline that lets you upload a PDF and ask natural-language questions against it, with the answer streamed token-by-token to your browser.

Most document Q&A tools send your files to OpenAI or another cloud service. This project keeps the entire pipeline on your machine: a MiniLM embedding model converts your document into semantic vectors, a pure-Python cosine-search index retrieves the most relevant paragraphs, and a 4-bit quantized Qwen2.5-3B-Instruct generates a grounded answer from those paragraphs alone.

There are no inference API calls and no database services. After the models are downloaded, no internet connection is required.

| File | Responsibility |
|---|---|
| `main.py` | Streamlit UI — chat interface, file upload, streaming output |
| `local_llm.py` | `AiModel` — loads the quantized Qwen model, builds the RAG prompt, streams token output |
| `local_embedding.py` | `LocalEmbedding` — MiniLM wrapper and vector index interface |
| `vector_index.py` | `VectorIndex` — pure-stdlib in-memory cosine/Euclidean vector store |
| `pdf_reader.py` | `PdfReader` — PDF text extraction and paragraph splitting |

---

## Screenshots

### Upload screen — ready state
<img width="1917" height="1018" alt="Screenshot 2026-10-08 141423" src="https://github.com/user-attachments/assets/9b5f639c-cf59-40d0-bdfc-028c08ff37bd" />


The sidebar shows **LLM ready** once the model has loaded. The PDF uploader accepts drag-and-drop or the file browser. The chat area waits for a document before accepting questions.

---

### Active conversation
![alt text](<Screenshot 2026-10-08 141423.png>)
After indexing, answers stream token-by-token into the chat. The sidebar displays the paragraph count for the indexed document. Follow-up questions reuse the cached index without re-embedding.

---

## Features

### Retrieval-Augmented Generation
- PDF ingestion with automatic paragraph extraction and whitespace normalisation
- Batch embedding of all paragraphs in a single GPU pass, with no per-paragraph round trips
- Cosine similarity search over 384-dimensional MiniLM vectors
- Configurable top-k retrieval to pass the most relevant context chunks to the LLM
- Strict grounding prompt — the model is instructed to answer only from the provided document text

### Efficient Local Inference
- 4-bit quantized Qwen2.5-3B-Instruct (`bitsandbytes`), roughly 2 GB on disk
- Runs on a consumer laptop GPU (tested on an RTX 3050 Laptop GPU) where the fp16 checkpoint exhausts VRAM and RAM

### Streaming Output
- `TextIteratorStreamer` runs `model.generate()` in a background daemon thread
- Tokens are yielded through the streamer queue without blocking the Streamlit main thread
- `st.write_stream()` renders tokens progressively as they arrive in the browser

### Caching and Session Management
- `@st.cache_resource` loads the LLM once per server process, so Streamlit reruns do not reload model weights
- `st.session_state` persists the embedding index across follow-up questions without re-indexing
- Uploading a new PDF resets the session and builds a fresh index automatically

### Pure-Python Vector Store
- `VectorIndex` uses no NumPy, FAISS, or external vector database
- Vectors are L2-normalised at index time, so similarity search reduces to a dot-product scan over stored vectors
- Both Euclidean and cosine metrics are available in the same class

---

## How It Works

1. **PDF → Paragraphs** — `PdfReader` reads every page, normalises whitespace, and splits on paragraph breaks, yielding a clean list of text chunks.

2. **Paragraphs → Vectors** — `LocalEmbedding.build_index()` batch-embeds all paragraphs in one GPU pass using `all-MiniLM-L6-v2` and stores the 384-dimensional L2-normalised vectors in `VectorIndex`.

3. **Question → Context** — at query time the question is embedded and compared against every stored vector by cosine distance. The top-k highest-scoring chunks are concatenated into a single context string.

4. **Context + Question → Answer** — `AiModel` wraps the context and question in a strict RAG prompt, launches `model.generate()` in a background daemon thread, and yields tokens through `TextIteratorStreamer` so `st.write_stream()` can render them progressively in the browser.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI (main.py)                │
│                                                          │
│  Sidebar: [Load Model]  [Upload PDF]  [Status]           │
│  Main:    [Chat input]  →  [Streamed answer]             │
└─────────────────────────┬────────────────────────────────┘
                          │
            ┌─────────────▼──────────────┐
            │        PDF Ingestion       │
            │  PdfReader: page text →    │
            │  clean paragraph list      │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │       LocalEmbedding       │
            │  all-MiniLM-L6-v2          │
            │  batch embed → 384-dim     │
            │  L2-normalised vectors     │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │        VectorIndex         │
            │  in-memory cosine store    │
            │  (pure Python stdlib)      │
            └─────────────┬──────────────┘
                          │ top-k chunks
            ┌─────────────▼──────────────┐
            │          AiModel           │
            │  Qwen2.5-3B-Instruct (4bit)│
            │  RAG prompt assembly       │
            │  model.generate() in thread│
            └─────────────┬──────────────┘
                          │ token stream
            ┌─────────────▼──────────────┐
            │    TextIteratorStreamer    │
            │  + st.write_stream()       │
            │  → live tokens in browser  │
            └────────────────────────────┘
```

**Model weight acquisition** (first run only):

```
HuggingFace Hub
  ├── sentence-transformers/all-MiniLM-L6-v2        → embedding model
  └── unsloth/Qwen2.5-3B-Instruct-bnb-4bit (~2 GB)  → generation model (cached locally)
```

---

## Project Structure

```
RAG_AI_SYSTEM/
├── main.py                  # Streamlit UI — entry point
├── local_llm.py             # AiModel: loads Qwen, orchestrates RAG, streams output
├── local_embedding.py       # LocalEmbedding: MiniLM wrapper + index interface
├── vector_index.py          # VectorIndex: pure-stdlib cosine/Euclidean vector store
├── pdf_reader.py            # PdfReader: PDF → clean paragraph list
├── requirements.txt         # Python dependencies (PyTorch installed separately)
├── pdfs/                    # Drop your PDFs here
├── application_screenshots/ # UI screenshots used in this README
├── .env                     # HF_TOKEN (gitignored — create this yourself)
└── .gitignore
```

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Earlier versions not tested |
| NVIDIA GPU with CUDA | Required for 4-bit quantized inference (`bitsandbytes`) |
| Hugging Face account | Free — needed for `HF_TOKEN` |

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/RAG_AI_SYSTEM.git
cd RAG_AI_SYSTEM

# Create and activate a virtual environment
python -m venv rag_env
source rag_env/Scripts/activate   # Windows (Git Bash)
# source rag_env/bin/activate     # macOS / Linux

# Install PyTorch (CUDA 12.6) first, separately
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install the remaining dependencies
pip install -r requirements.txt
```

> **GPU note:** the `cu126` wheel targets CUDA 12.6. Find the right wheel for your GPU at [pytorch.org/get-started](https://pytorch.org/get-started/locally/). PyTorch is installed separately because CUDA-specific wheels (e.g. `torch+cu126`) break a plain `pip install -r requirements.txt`.

### Add your Hugging Face token

Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Read access is sufficient. Never commit this file — it is listed in `.gitignore`.

---

## Usage

```bash
streamlit run main.py
```

Streamlit will print a local URL (default `http://localhost:8501`). Open it in your browser.

> **First run:** both models are downloaded from the Hugging Face Hub and cached in `~/.cache/huggingface/`. This takes a few minutes depending on your connection. Subsequent runs start in seconds.

### Basic workflow

1. Wait for **"LLM ready"** in the sidebar — the model has finished loading.
2. Drag and drop any PDF onto the uploader, or click **Browse**.
3. Wait for **"Document ready — N paragraphs indexed"** in the sidebar.
4. Type your question in the chat input at the bottom and press Enter.
5. The answer streams token-by-token. Ask follow-up questions freely — the index is cached.
6. Upload a new PDF to start a fresh conversation.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **UI** | Streamlit | Chat interface, file upload, live streaming |
| **LLM** | Qwen2.5-3B-Instruct (4-bit, bitsandbytes) | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim semantic search vectors |
| **Inference** | Hugging Face Transformers | Model loading and generation |
| **Compute** | PyTorch (CUDA 12.6) | GPU-accelerated inference |
| **PDF parsing** | pypdf | Text extraction |
| **Vector store** | Custom `VectorIndex` | In-memory cosine search, no external DB |
| **Streaming** | `TextIteratorStreamer` | Non-blocking token delivery to UI |
| **Config** | python-dotenv | `.env`-based HF token loading |
| **Hub access** | huggingface_hub | Model download and authentication |

---

## Limitations

- **Single document at a time** — the index holds one PDF; there is no multi-document or cross-document Q&A.
- **In-memory index only** — `VectorIndex` is not persisted to disk, so re-uploading the same PDF re-embeds it from scratch.
- **Linear scan** — similarity search scans every stored vector, so performance degrades on very large documents with thousands of paragraphs.
- **Paragraph chunking only** — the splitter uses whitespace-based paragraph breaks; fixed-token sliding-window chunking is not implemented.
- **Context window cap** — the top-k chunks must fit within the model's context window; very long paragraphs or a high `k` can exceed it.
- **GPU required** — 4-bit inference through `bitsandbytes` needs an NVIDIA GPU.
- **Cold start** — the first run downloads both models from the Hugging Face Hub, which can take several minutes.
- **Single-turn grounding** — the RAG prompt is rebuilt from scratch on every question; no conversation memory is carried across turns.

---

## Future Improvements

- **Persistent index** — serialize `VectorIndex.vectors` and `.documents` to disk so documents are not re-embedded on every startup.
- **Multi-document support** — merge indices from several PDFs into one `VectorIndex` for cross-document Q&A.
- **Sliding-window chunking** — replace paragraph splits with fixed-token overlapping windows for more uniform chunks and better boundary handling.
- **ANN indexing** — replace linear scan with an approximate nearest-neighbour structure (e.g. HNSW) for sub-linear search at scale.
- **Larger LLM support** — swap the model name in `AiModel.__init__` for any other Hugging Face causal model, using 4-bit quantization to fit larger models on small GPUs.
- **Conversation memory** — accumulate prior Q&A turns in the RAG prompt for contextually aware follow-ups.
- **Embedding progress bar** — show per-paragraph indexing progress during PDF ingestion instead of a single blocking wait.
