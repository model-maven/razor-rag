"""Core RAG logic: retrieval from Pinecone + streaming answers from Claude."""
from config import INDEX_NAME, EMBEDDING_MODEL, CLAUDE_MODEL, TOP_K

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided document excerpts.

Rules:
- Base your answer strictly on the excerpts. If they don't contain the answer, say so plainly.
- Cite your sources inline like [source.pdf, p. 3] after the relevant statement.
- Be concise and direct."""


def retrieve(pc, index, question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the question and fetch the most similar chunks."""
    embedding = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[question],
        parameters={"input_type": "query"},
    )[0]["values"]

    result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [
        {
            "score": round(m["score"], 3),
            "source": m["metadata"].get("source", "?"),
            "page": m["metadata"].get("page", "?"),
            "text": m["metadata"].get("text", ""),
        }
        for m in result["matches"]
    ]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"<excerpt id=\"{i}\" source=\"{c['source']}\" page=\"{c['page']}\">\n{c['text']}\n</excerpt>"
        )
    return "\n\n".join(parts)


def stream_answer(claude, question: str, chunks: list[dict], history: list[dict]):
    """Yield answer text deltas from Claude as they arrive."""
    context = build_context(chunks)
    messages = history + [{
        "role": "user",
        "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}",
    }]
    with claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
