"""
FastAPI backend for the PDF RAG assistant.

Dev:   uvicorn main:app --reload --port 8000
Prod:  build the frontend (npm run build) and this server also serves it.
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import get_clients, INDEX_NAME, RAG_ENGINE

if RAG_ENGINE == "langgraph":
    from rag_langgraph import retrieve, stream_answer
elif RAG_ENGINE == "langchain":
    from rag_langchain import retrieve, stream_answer
else:
    from rag import retrieve, stream_answer
print(f"RAG engine: {RAG_ENGINE}")

# ---------- Optional password protection ----------
# Set APP_USERNAME and APP_PASSWORD (env vars / .env) to require a login.
# Leave them unset to keep the app open (e.g. local development).
import base64
import os
import secrets as _secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
AUTH_ENABLED = bool(APP_USERNAME and APP_PASSWORD)
print(f"Password protection: {'ON' if AUTH_ENABLED else 'off'}")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        header = request.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8")
                username, _, password = decoded.partition(":")
                # constant-time comparison to prevent timing attacks
                user_ok = _secrets.compare_digest(username, APP_USERNAME)
                pass_ok = _secrets.compare_digest(password, APP_PASSWORD)
                if user_ok and pass_ok:
                    return await call_next(request)
            except Exception:
                pass
        return Response(
            "Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Document Assistant"'},
        )

app = FastAPI(title="PDF RAG API")

if AUTH_ENABLED:
    app.add_middleware(BasicAuthMiddleware)

# Allow the Vite dev server to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Clients (created once at startup) ----------
pc, claude = get_clients()
_index = None


def get_index():
    global _index
    if _index is None:
        existing = [ix["name"] for ix in pc.list_indexes()]
        if INDEX_NAME in existing:
            _index = pc.Index(INDEX_NAME)
    return _index


# ---------- Schemas ----------
class HistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = []


# ---------- API routes ----------
@app.get("/api/health")
def health():
    try:
        index = get_index()
        if index is None:
            return {"status": "no_index", "detail": f"Index '{INDEX_NAME}' not found. Run ingest.py."}
        stats = index.describe_index_stats()
        return {"status": "ok", "vectors": stats.get("total_vector_count", 0)}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Streams newline-delimited JSON events:
    {"type":"sources","sources":[...]} then {"type":"token","text":"..."} ... {"type":"done"}
    """
    index = get_index()

    def event(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def generate():
        if index is None:
            yield event({"type": "error", "detail": "No document index. Run ingest.py first."})
            return
        try:
            chunks = retrieve(pc, index, req.message)
            yield event({"type": "sources", "sources": chunks})
            if not chunks:
                yield event({"type": "token", "text": "I couldn't find anything relevant in the documents for that question."})
                yield event({"type": "done"})
                return
            # last 3 exchanges of history, capped
            history = [m.model_dump() for m in req.history][-6:]
            for delta in stream_answer(claude, req.message, chunks, history):
                yield event({"type": "token", "text": delta})
            yield event({"type": "done"})
        except Exception as exc:  # surface errors to the UI instead of hanging
            yield event({"type": "error", "detail": str(exc)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------- Serve the built frontend in production ----------
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        return FileResponse(DIST / "index.html")
