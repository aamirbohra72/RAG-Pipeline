# Chat with your PDFs — Senior RAG

Service-oriented Retrieval-Augmented Generation with:
- JWT auth + per-user document isolation
- Hybrid retrieval + **cross-encoder re-ranker**
- **OCR** for scanned / image-only PDFs
- Streaming answers (SSE)

## Architecture

```
backend/
├── main.py
├── scripts/make_scanned_pdf.py   # builds a scan-like PDF for OCR tests
├── app/
│   ├── routers/   # HTTP
│   └── services/
│       ├── pdf_service.py      # pypdf + RapidOCR fallback
│       ├── retrieval.py        # hybrid fusion
│       ├── rerank_service.py   # cross-encoder / ms-marco MiniLM
│       ├── rag_service.py
│       └── ...
```

### Retrieval pipeline (now)

```
question
  → embed (mistral-embed)
  → Chroma top-N (user-scoped)
  → hybrid score (vector + lexical)
  → cross-encoder re-rank → top_k
  → prompt mistral-large → answer + citations
```

### Ingest pipeline (now)

```
PDF
  → pypdf text layer
  → if page < OCR_MIN_CHARS chars → render page → RapidOCR
  → chunk → batch embed → Chroma (+ user_id)
```

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # set MISTRAL_API_KEY (+ JWT_SECRET)
uvicorn main:app --reload --port 8001
```

```bash
cd frontend
npm install
# .env.local → NEXT_PUBLIC_API_URL=http://localhost:8001
npm run dev
```

First query after install downloads the cross-encoder + OCR ONNX models (one-time, may take a few minutes).

## How to verify

### 0. Health flags

```bash
curl http://localhost:8001/health
```

Expect `"rerank_enabled": true` and `"ocr_enabled": true`.

### 1. Cross-encoder re-ranker

1. Log in on http://localhost:3000 and upload a normal text PDF (e.g. company handbook).
2. Ask a specific question: `What is the unique passphrase / vacation policy?`
3. In the Sources panel you should see **`rerank …`** scores (cross-encoder), not only hybrid scores.
4. Optional API check (after login token):

```bash
curl -s -X POST http://localhost:8001/query ^
  -H "Authorization: Bearer YOUR_TOKEN" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How many vacation days?\"}"
```

Look at `sources[].rerank_score`. Toggle off to compare:

```
# in backend/.env
RERANK_ENABLED=false
```

Restart uvicorn — sources then use hybrid `score` only (frontend falls back).

### 2. OCR for scanned PDFs

Create an image-only PDF (no text layer):

```bash
cd backend
venv\Scripts\activate
python scripts/make_scanned_pdf.py
```

Upload `sample_scanned_handbook.pdf` via the UI.

- Upload API response / logs should show `"ocr_pages": 1` (and `"text_pages": 0`).
- Ask: `What is the unique passphrase in the memo?`
- Expect answer mentioning **BLUE-ORBIT-77**.

If OCR is off:

```
OCR_ENABLED=false
```

Re-upload the scanned PDF → should fail with “No extractable text” (proves OCR was doing the work).

## Tunable env vars

| Var | Default | Meaning |
|-----|---------|---------|
| `RERANK_ENABLED` | true | Cross-encoder on/off |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HF model id |
| `CANDIDATE_POOL` | 20 | Hybrid shortlist size before re-rank |
| `TOP_K` | 4 | Chunks sent to the LLM |
| `OCR_ENABLED` | true | OCR fallback on/off |
| `OCR_MIN_CHARS` | 40 | Below this → treat page as scanned |
| `OCR_DPI` | 200 | Render resolution for OCR |

## Notes / gotchas

- Re-ranker first load downloads ~80MB+ of weights; keep the process warm.
- OCR is slower than digital text — expected for scans.
- Old chunks without `user_id` stay invisible after auth — re-upload under your account.
- Prefer backend on **8001** if 8000 still runs an old process.
