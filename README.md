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
