# Document Q&A RAG Agent (Streamlit + FastAPI + LangChain + LangGraph + Groq)

A working usecase that lets you upload documents (PDF / TXT / DOCX),
indexes them into a vector store, and answers natural-language questions
about them — grounded in the documents, with source, page, and chunk
references. Two ways to use it: a Streamlit chat UI, and a small FastAPI
backend for programmatic access.

## How it works

```
Upload → Parse → Clean → Chunk (tagged with chunk_index) → Embed → Store in Chroma
                                                                        │
User question ──────────────────────────────────────────────────────────┘
        │
        ▼
   LangGraph workflow
   ┌──────────────────────────────────────────────────────────────────────┐
   │ rewrite (resolve follow-ups using chat history) →                    │
   │   retrieve → grade (filter irrelevant chunks) →                      │
   │     ├─ no relevant chunks → "not found in documents" → END           │
   │     └─ generate → check groundedness →                               │
   │           ├─ grounded / out of retries → END                        │
   │           └─ not grounded & retries left → generate (retry)          │
   └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
   Answer + source document / page / chunk references
   (sources are omitted entirely when the answer is the "not found" fallback)
```

- **LLM**: Groq (`openai/gpt-oss-120b` by default, selectable in the
  Streamlit sidebar) via `langchain-groq`. Note: `llama-3.3-70b-versatile`
  and `llama-3.1-8b-instant` were deprecated by Groq on 2026-08-16 for
  free/developer-tier accounts — this app uses their current production
  replacements instead.
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` via
  `langchain-huggingface` — no extra API key needed, runs on CPU.
- **Vector store**: Chroma, persisted to disk per session (thread id).
  Each chunk is tagged with its source filename, PDF page (when
  applicable), and its position within the file (`chunk_index`).
- **Orchestration**: LangGraph `StateGraph` with a history-aware rewrite
  step, conditional routing for insufficient retrieval, and a
  groundedness retry loop before falling back to a safe "not found"
  answer.
- **API layer**: a minimal FastAPI app (`api.py`) exposing the same
  pipeline over HTTP, for use outside the Streamlit UI.

## Project structure

```
document-qa-rag-agent/
├── app.py                  # Streamlit UI (entry point)
├── api.py                   # FastAPI backend (entry point)               
├── requirements.txt
├── .env
├── src/
│   ├── config.py           # settings (env vars + defaults)
│   ├── ingestion.py         # FR-01/02/03: load, parse, chunk documents (tags chunk_index)
│   ├── vectorstore.py       # FR-04/05/06: embeddings + Chroma + retriever
│   ├── state.py             # LangGraph state schema (includes chat_history)
│   ├── graph.py              # FR-09: LangGraph workflow (rewrite/retrieve/grade/generate/check)
│   └── utils.py              # FR-11: logging helper
└── data/
    ├── uploads/              # uploaded files land here, per session
    └── vectorstore/          # Chroma persistence, per session
```

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   cd document-qa-rag-agent
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get a Groq API key** — sign up at https://console.groq.com (free
   tier available).

3. **Configure the key** — copy `.env.example` to `.env` and fill in
   `GROQ_API_KEY`. The key is read only from this server-side file; it
   is **not** entered anywhere in the browser or accepted in any API
   request, so it can't be exposed by screen shares, browser extensions,
   or a shared/public deployment.

## Running the Streamlit app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Upload a
PDF/TXT/DOCX, click "Index Documents," then ask questions in the chat
box. Each answer shows an expandable "Sources" section with the file,
page (PDFs only), and chunk number it came from — omitted entirely when
the answer is the "not found" fallback, so you never see a citation for
content that wasn't actually used.

## Running the API

`api.py` wraps the same `src/` pipeline behind a small FastAPI service,
for use from a frontend, curl, Postman, or another backend instead of
(or alongside) the Streamlit UI.

```bash
uvicorn api:app --reload
```

Interactive docs at http://localhost:8000/docs. Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check; also reports whether a Groq key is configured (never the key itself). |
| POST | `/documents/upload` | Upload **one file** per call (`file` field). Pass `thread_id` (form field) to add to an existing session, or omit it to start a new one — the response returns the `thread_id` to reuse. Renders as a single, unambiguous file-picker in Swagger UI. |
| POST | `/chat` | JSON body `{"question": "...", "thread_id": "..."}` → `{"answer": ..., "sources": [{"source", "page", "chunk"}, ...]}`. 

**To index multiple documents via the API**, call `/documents/upload`
once per file, reusing the `thread_id` from the first response on every
subsequent call.

**Security note:** the Groq API key is read once from the server's
`.env` file (`Config.GROQ_API_KEY`) and is never accepted as a request
field on any endpoint above.


TC-03/04/08 (answer quality, groundedness, hallucination avoidance) and
TC-05 (follow-up questions) depend on real LLM output quality, which is
better verified by manual sessions with a real Groq key than by a
brittle automated assertion on generated text.

## Requirements coverage (from the POC brief)

| ID | Requirement | Implementation |
|----|-------------|-----------------|
| FR-01 | Document upload (PDF/TXT/DOCX) | Streamlit file uploader; FastAPI `/documents/upload` (one file per call) |
| FR-02 | Text extraction, graceful errors on empty/invalid docs | `src/ingestion.py` |
| FR-03 | Configurable chunking with overlap, chunk position tagging | `src/ingestion.py`, sidebar sliders |
| FR-04 | Embeddings | `src/vectorstore.py` (HuggingFace, local) |
| FR-05 | Vector store (Chroma) | `src/vectorstore.py` |
| FR-06 | Top-k semantic retrieval | `src/vectorstore.py` / `graph.py` retrieve node |
| FR-07 | Answer generation grounded in context | `graph.py` generate node |
| FR-08 | Source/page/chunk metadata in response | `graph.py` generate node, `app.py` Sources expander, `api.py` `/chat` response |
| FR-09 | LangGraph orchestration | `src/graph.py` |
| FR-10 | Session/thread id for follow-ups | `app.py` (`st.session_state.thread_id`), `api.py` (`thread_id` field) |
| FR-11 | Error handling & logging | `src/ingestion.py` exceptions, `src/utils.py`, try/except in `app.py` and `api.py` |

## Notes & known limitations

- **Follow-up questions (TC-05)** are resolved by a rewrite step
  (`src/graph.py` → `rewrite_question_node`) that runs before retrieval,
  using recent chat history to turn a dependent question like "what
  about the second one?" into a standalone one. Skipped entirely (no
  extra LLM call) when there's no prior history yet.
- **Sources only appear when they were actually used**: if the LLM
  declines to answer (the "I could not find this information..."
  fallback), `sources` is returned as an empty list rather than listing
  chunks that were retrieved but not actually relied on.
- **Ambiguous multi-document questions** (e.g. "what does this document
  say" with more than one file indexed) can return inconsistent results
  between runs, since the retriever has no way to know which document
  "this" refers to — worth a UI-level disambiguation prompt if this
  matters for your use case.
- **Retrieval precision gap observed**: a direct-definition question for
  a term that only appears in passing (not as a self-contained
  definition) elsewhere in an indexed document can still return a
  not-found response — worth investigating whether chunking or grading
  is dropping passing mentions.
- Vector store and embedding model are swappable — moving to Pinecone or
  a hosted embeddings API only touches `vectorstore.py`.
- Multi-user / production deployment would need persistent storage
  outside the local filesystem and authentication in front of the app.
- Chunks indexed **before** the `chunk_index` field was added won't have
  it in their stored metadata — re-upload (or clear the session with
  `DELETE /documents/{thread_id}` / the Streamlit button) to backfill it.