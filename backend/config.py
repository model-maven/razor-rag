"""Shared configuration: loads .env and creates API clients."""
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone
from anthropic import Anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
EMBEDDING_DIM = 1024  # dimension of multilingual-e5-large
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Chunking settings
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters of overlap between chunks
TOP_K = 5              # how many chunks to retrieve per question

# Engine switch: "plain" | "langchain" | "langgraph"
# (USE_LANGCHAIN=true still works as a shortcut for "langchain")
RAG_ENGINE = os.getenv("RAG_ENGINE", "").strip().lower()
if not RAG_ENGINE:
    _legacy = os.getenv("USE_LANGCHAIN", "false").strip().lower() in {"1", "true", "yes"}
    RAG_ENGINE = "langchain" if _legacy else "plain"
if RAG_ENGINE not in {"plain", "langchain", "langgraph"}:
    RAG_ENGINE = "plain"


def check_keys():
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def get_clients():
    check_keys()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    claude = Anthropic(api_key=ANTHROPIC_API_KEY)
    return pc, claude
