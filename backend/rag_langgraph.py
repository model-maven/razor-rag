"""
LangGraph version of the RAG logic: self-correcting retrieval.

The retrieval step is a small state graph instead of a single call:

    retrieve ──► grade relevance ──► (good, or 2 attempts) ──► done
                      │
                      └─► rewrite question ──► retrieve (again)

If the first search returns weak chunks, an LLM judges them, rewrites the
question into better search terms, and retries once. Answer generation is
then streamed exactly like the other engines.

Exposes the SAME interface as rag.py / rag_langchain.py:
    retrieve(pc, index, question)  -> list[chunk dicts]
    stream_answer(claude, question, chunks, history) -> yields text
so main.py can switch engines freely.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from config import TOP_K
from rag import SYSTEM_PROMPT, build_context
from rag_langchain import _llm, retrieve as vector_search


# ---------- Graph state ----------
class RAGState(TypedDict):
    question: str            # original customer question (never changes)
    search_query: str        # what we actually search with (may get rewritten)
    chunks: list[dict]
    attempts: int
    pc: object
    index: object


# ---------- Nodes ----------
def retrieve_node(state: RAGState) -> dict:
    chunks = vector_search(state["pc"], state["index"], state["search_query"], top_k=TOP_K)
    return {"chunks": chunks, "attempts": state["attempts"] + 1}


def grade_node(state: RAGState) -> dict:
    """No state change; the decision happens in the conditional edge below."""
    return {}


def decide(state: RAGState) -> str:
    """Are the retrieved chunks relevant enough to answer from?"""
    if not state["chunks"]:
        return "rewrite" if state["attempts"] < 2 else "done"
    if state["attempts"] >= 2:
        return "done"  # one retry max — never loop forever

    preview = "\n---\n".join(c["text"][:300] for c in state["chunks"][:3])
    verdict = _llm().invoke(
        [HumanMessage(
            "You are grading search results.\n\n"
            f"Question: {state['question']}\n\n"
            f"Retrieved passages:\n{preview}\n\n"
            "Do these passages contain information that helps answer the question? "
            "Reply with exactly one word: yes or no."
        )],
        max_tokens=5,
    )
    answer = verdict.content if isinstance(verdict.content, str) else str(verdict.content)
    return "done" if "yes" in answer.lower() else "rewrite"


def rewrite_node(state: RAGState) -> dict:
    """Rephrase the question into better search terms and try again."""
    result = _llm().invoke(
        [HumanMessage(
            "The following question was used to search a document database but "
            "returned poor results. Rewrite it as a better search query: use "
            "likely document vocabulary, expand abbreviations, remove filler. "
            "Reply with ONLY the rewritten query.\n\n"
            f"Question: {state['question']}"
        )],
        max_tokens=100,
    )
    new_query = result.content if isinstance(result.content, str) else state["question"]
    return {"search_query": new_query.strip()}


# ---------- Build the graph (compiled once at import) ----------
_builder = StateGraph(RAGState)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("grade", grade_node)
_builder.add_node("rewrite", rewrite_node)
_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "grade")
_builder.add_conditional_edges("grade", decide, {"done": END, "rewrite": "rewrite"})
_builder.add_edge("rewrite", "retrieve")
GRAPH = _builder.compile()


# ---------- Public interface (same as the other engines) ----------
def retrieve(pc, index, question: str, top_k: int = TOP_K) -> list[dict]:
    final_state = GRAPH.invoke({
        "question": question,
        "search_query": question,
        "chunks": [],
        "attempts": 0,
        "pc": pc,
        "index": index,
    })
    return final_state["chunks"]


def stream_answer(claude, question: str, chunks: list[dict], history: list[dict]):
    """Stream the final answer (same generation as the LangChain engine)."""
    messages = [("system", SYSTEM_PROMPT)]
    for m in history:
        messages.append(
            HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        )
    messages.append(HumanMessage(
        f"Document excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    ))
    for chunk in _llm().stream(messages):
        content = chunk.content
        if isinstance(content, str):
            if content:
                yield content
        else:
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                if text:
                    yield text
