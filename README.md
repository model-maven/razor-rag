# PDF RAG — FastAPI + React

A customer-facing document Q&A app. Architecture:

```
Browser (React)  ──►  FastAPI backend  ──►  Pinecone (search)
                                       ──►  Claude (answers, streamed)
```

```
rag-app/
├── backend/
│   ├── main.py            # FastAPI: /api/health, /api/chat (streaming)
│   ├── rag.py             # retrieval + Claude logic
│   ├── config.py          # env vars, clients, tunable settings
│   ├── ingest.py          # PDF → chunks → embeddings → Pinecone (CLI)
│   ├── data/              # drop PDFs here
│   ├── requirements.txt
│   └── .env.example       # copy to .env, add your two API keys
└── frontend/
    ├── src/               # React app (chat UI, streaming, source cards)
    ├── index.html
    ├── vite.config.js     # proxies /api → backend in dev
    └── package.json
```

## Prerequisites

- Python 3.10+
- Node.js 18+ (https://nodejs.org — needed to build/run the React frontend)
- API keys: Anthropic (console.anthropic.com) and Pinecone (app.pinecone.io)

## Setup

**Backend** (terminal 1):

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit .env and paste your keys
```

Put PDFs in `backend/data/` and ingest them:

```bash
python ingest.py
```

Start the API:

```bash
python -m uvicorn main:app --reload --port 8000
```

**Frontend** (terminal 2):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — that's the customer-facing app. The header shows a live status ("Connected · N passages indexed"), answers stream in as they're generated, and each answer shows clickable source cards with page numbers and excerpts.

## Production (one server)

Build the frontend once, and FastAPI will serve it — no Node needed on the server:

```bash
cd frontend && npm run build      # creates frontend/dist
cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Now http://your-server:8000 serves the whole app (UI + API together). Deploy this on Render, Railway, Fly.io, or any VPS. Set `ANTHROPIC_API_KEY` and `PINECONE_API_KEY` as environment variables on the host instead of shipping a `.env` file.

Before real customers, also add:
- **HTTPS + a domain** (Render/Railway give you this automatically)
- **Authentication** if the documents aren't public
- **Rate limiting** (e.g. `slowapi`) so one user can't run up your Claude bill
- **Lock down CORS** in `main.py` to your real domain

## API reference

- `GET /api/health` → `{"status":"ok","vectors":1234}`
- `POST /api/chat` with `{"message":"...", "history":[{"role":"user","content":"..."}]}` → streams newline-delimited JSON: a `sources` event, then `token` events, then `done`.

## Three RAG engines: plain, LangChain, LangGraph

The backend ships with **three interchangeable engines**, selected in `.env`:

```
RAG_ENGINE=plain        # or: langchain | langgraph
```

Restart the backend after changing it — it prints the active engine on startup.

| Engine | Files | What it does |
|---|---|---|
| `plain` (default) | `rag.py`, `ingest.py` | Direct SDKs: pypdf + pinecone + anthropic. Simplest to read and debug. |
| `langchain` | `rag_langchain.py`, `ingest_langchain.py` | Same pipeline built from LangChain components (`PineconeVectorStore`, `ChatAnthropic`, LCEL). |
| `langgraph` | `rag_langgraph.py` | **Self-correcting retrieval**: a state graph that retrieves, grades the results with an LLM, and — if they look irrelevant — rewrites the question and retries once before answering. |

The LangGraph flow:

```
question → retrieve → grade relevance ──ok──► answer (streamed)
                          │
                        weak
                          ▼
                  rewrite question → retrieve again → answer
```

All three write to and read from the **same Pinecone index with the same metadata**, and expose the same `retrieve()` / `stream_answer()` interface — the frontend never knows which engine is running. Trade-off to know: `langgraph` can cost 1–2 extra small LLM calls per question (the grader, and the rewriter when triggered), so answers are slightly slower but more robust to badly-phrased questions.

**Ingesting with LangChain instead of the plain script:**

```bash
python ingest_langchain.py
```

Safe alongside `ingest.py` — chunk IDs are deterministic in both, so re-ingesting overwrites rather than duplicates.

## Password protection

Set two environment variables to require a login for the entire app (page + API):

```
APP_USERNAME=amir
APP_PASSWORD=a-strong-password
```

Locally: add them to `.env`. On Railway: add them in the Variables tab. The browser
shows a native username/password prompt on first visit and remembers it for the
session. Remove the variables to turn the login off. The backend prints
`Password protection: ON` at startup when active.

## Tuning

`backend/config.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `CLAUDE_MODEL`.

## Troubleshooting

- **"Backend offline" in the header** — the FastAPI server isn't running, or it's on a different port than 8000.
- **"No documents indexed yet"** — run `python ingest.py` in `backend/` with PDFs in `backend/data/`.
- **`npm` not found** — install Node.js from nodejs.org, then reopen the terminal.
- **CORS errors in the browser console** — you changed ports; update `allow_origins` in `main.py` and the proxy in `vite.config.js`.
