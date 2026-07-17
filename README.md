# Chat with your PDFs — Architecture + LangGraph / LangSmith

## Current architecture (what you have now)

```mermaid
flowchart TB
  subgraph Client
    UI[Next.js Frontend]
  end

  subgraph API["FastAPI Backend"]
    Auth[JWT Auth]
    Upload[Upload + OCR]
    Query[Query / Stream]
  end

  subgraph Orchestration["LangGraph"]
    R[retrieve node]
    G[generate node]
    E[empty node]
    R -->|docs found| G
    R -->|no docs| E
  end

  subgraph RetrieveStack["Inside retrieve"]
    Emb[MistralAIEmbeddings]
    VS[(Neon pgvector / Chroma)]
    Hyb[Hybrid fusion]
    RR[Cross-encoder re-rank]
    Emb --> VS --> Hyb --> RR
  end

  subgraph GenerateStack["Inside generate"]
    Prompt[ChatPromptTemplate]
    LLM[ChatMistralAI]
    Prompt --> LLM
  end

  subgraph Observe["LangSmith"]
    Trace[Traces / latency / tokens]
  end

  UI -->|Bearer token| Auth
  UI --> Upload
  UI --> Query
  Upload -->|chunks + embeddings| VS
  Query --> R
  R --> RetrieveStack
  G --> GenerateStack
  R -.->|traced| Trace
  G -.->|traced| Trace
  LLM -.->|traced| Trace
```

### Request lifecycle (simple)

1. **Register / login** → JWT  
2. **Upload PDF** → text or OCR → chunk → embed → **Neon `document_chunks`** (scoped by `user_id`)  
3. **Ask question** → LangGraph `retrieve` (vector + hybrid + re-rank) → `generate` (Mistral) → stream answer + citations  
4. **LangSmith** (optional key) records each run for debugging  

---

## LangGraph + LangSmith — yes, possible (now wired)

| Tool | Role in this project |
|------|----------------------|
| **LangGraph** | Explicit graph: `retrieve → generate` (or `empty`) instead of a flat LCEL-only path |
| **LangSmith** | Cloud monitor: traces, latency, token usage, failed steps |

They do **not** replace Neon or auth. They sit on top of your existing RAG.

### Target architecture (with monitoring)

```mermaid
flowchart LR
  subgraph App
    FE[Frontend]
    BE[FastAPI]
    LG[LangGraph RAG]
  end

  subgraph Data
    Neon[(Neon + pgvector)]
    Users[(SQLite users.db)]
  end

  subgraph CloudAI
    Mistral[Mistral embed + chat]
    LS[LangSmith UI]
  end

  FE --> BE
  BE --> Users
  BE --> LG
  LG -->|retrieve| Neon
  LG -->|embed/chat| Mistral
  LG -->|traces| LS
```

---

## Setup LangSmith

1. Create account: https://smith.langchain.com  
2. Create an API key  
3. In `backend/.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=genai-rag
```

4. Restart uvicorn  

5. Ask a question in the UI → open LangSmith → project **genai-rag** → see runs for retriever + ChatMistralAI / LangGraph  

Without a key, the app still works; `/health` shows `"langsmith_tracing": false`.

---

## Validate

```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

| Check | Expect |
|--------|--------|
| `GET /health` | `langgraph_enabled: true`, `langchain.orchestration: langgraph` |
| Logs on startup | `LangGraph RAG compiled` |
| With API key | `LangSmith tracing ON` + runs in smith.langchain.com |
| Ask a question | Answer + sources unchanged; LangSmith shows retrieve then generate |
| No docs | Graph takes `empty` path → “No documents have been uploaded yet.” |

---

## Config summary

```env
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=genai-rag
```

Toggle tracing off anytime with `LANGSMITH_TRACING=false`.

---

## Async PDF ingest — Celery + Upstash Redis

Uploading a PDF triggers heavy work (OCR → chunk → embed → store). Doing it
inline blocks the HTTP request and can time out on big/scanned files. Instead we
push the work to a **Celery** worker using **Upstash Redis** as the broker +
result backend. The API returns instantly with a job id; the frontend polls for
progress.

```mermaid
flowchart LR
  FE[Frontend] -->|POST /upload/async| API[FastAPI]
  API -->|save temp PDF + enqueue| Redis[(Upstash Redis broker)]
  API -->|job_id| FE
  Worker[Celery worker] -->|BRPOP job| Redis
  Worker -->|extract → chunk → embed| Mistral[Mistral embed]
  Worker -->|store chunks| Neon[(Neon pgvector)]
  Worker -->|state/result| Redis
  FE -->|GET /jobs/id poll| API
  API -->|AsyncResult| Redis
```

### Flow

1. `POST /upload/async` — saves each PDF to `./uploads/{uuid}.pdf`, enqueues an
   `ingest.process_pdf` task, returns `{ jobs: [{ job_id, filename }] }`.
2. Worker runs the pipeline, reporting `PROGRESS` states
   (`reading → extracting → chunking → embedding`) and deletes the temp file.
3. `GET /jobs/{job_id}` — returns `state` (`PENDING`/`PROGRESS`/`SUCCESS`/`FAILURE`),
   current `stage`, and on success the `UploadItem` result (chunks, OCR pages).

The original synchronous `POST /upload` still exists as a fallback.

### Config (`backend/.env`)

```env
ASYNC_INGEST=true
# rediss:// = TLS (required by Upstash). /0 = db index.
REDIS_URL=rediss://default:YOUR_UPSTASH_TOKEN@YOUR-DB.upstash.io:6379/0
```

### Run the worker (Windows)

RQ needs `os.fork`, which Windows lacks, so we use Celery with the **solo** pool.
Open a **second terminal** alongside uvicorn:

```powershell
cd backend
venv\Scripts\activate
celery -A app.celery_app.celery worker --pool=solo -l info
```

On Linux/macOS you can use the default prefork pool (drop `--pool=solo`).

### Validate

| Check | Expect |
|-------|--------|
| `GET /health` | `"async_ingest": true`, `"redis_ok": true` |
| Worker startup log | `[tasks] . ingest.process_pdf` |
| Upload a PDF in UI | returns immediately; a **Processing** row shows `extracting… → embedding…` |
| Worker terminal | logs `Async ingest <file>: ... chunks=N` |
| `GET /jobs/{id}` | `state` transitions to `SUCCESS` with `result.chunks` |
| Stop the worker, upload again | job stays `PENDING` (proof the queue is decoupled) |

> **Upstash note:** the free tier bills per command and caps connections.
> The Celery config keeps usage modest (`broker_pool_limit=2`,
> `worker_prefetch_multiplier=1`). Rotate the token you shared in chat.
