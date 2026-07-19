"""
LangChain version of the RAG logic.

Exposes the SAME interface as rag.py — retrieve(...) and stream_answer(...) —
so main.py can swap between the two with the USE_LANGCHAIN setting.

Components used:
  PineconeEmbeddings   -> embeds the question (Pinecone-hosted model, no extra key)
  PineconeVectorStore  -> similarity search retriever
  ChatAnthropic        -> Claude, streamed
  ChatPromptTemplate   -> assembles the grounded prompt
"""
from functools import lru_cache

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    ANTHROPIC_API_KEY,
    PINECONE_API_KEY,
    EMBEDDING_MODEL,
    CLAUDE_MODEL,
    TOP_K,
)
from rag import SYSTEM_PROMPT, build_context  # same prompt & context format


@lru_cache(maxsize=1)
def _embeddings():
    return PineconeEmbeddings(model=EMBEDDING_MODEL, pinecone_api_key=PINECONE_API_KEY)


@lru_cache(maxsize=1)
def _llm():
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        api_key=ANTHROPIC_API_KEY,
    )


PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{history}"),
    ("human", "Document excerpts:\n\n{context}\n\nQuestion: {question}"),
])


def retrieve(pc, index, question: str, top_k: int = TOP_K) -> list[dict]:
    """Similarity search via LangChain's vector store abstraction."""
    store = PineconeVectorStore(index=index, embedding=_embeddings(), text_key="text")
    results = store.similarity_search_with_score(question, k=top_k)
    return [
        {
            "score": round(float(score), 3),
            "source": doc.metadata.get("source", "?"),
            "page": doc.metadata.get("page", "?"),
            "text": doc.page_content,
        }
        for doc, score in results
    ]


def stream_answer(claude, question: str, chunks: list[dict], history: list[dict]):
    """Yield answer text deltas. (`claude` arg is unused here — kept for interface parity.)"""
    lc_history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in history
    ]
    chain = PROMPT | _llm()
    for chunk in chain.stream({
        "history": lc_history,
        "context": build_context(chunks),
        "question": question,
    }):
        content = chunk.content
        if isinstance(content, str):
            if content:
                yield content
        else:  # content blocks
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                if text:
                    yield text
