"""
LangChain version of the ingestion pipeline.

Same job as ingest.py (PDF -> chunks -> embeddings -> Pinecone), but built
from LangChain components:

  PyPDFLoader  ->  RecursiveCharacterTextSplitter  ->  PineconeVectorStore

It writes to the SAME index with the SAME metadata (source, page, text),
so ingest.py and this script are interchangeable.

Usage:
    python ingest_langchain.py                  # every PDF in ./data
    python ingest_langchain.py path/to/file.pdf
"""
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

from config import (
    get_clients,
    PINECONE_API_KEY,
    INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from ingest import ensure_index  # reuse index creation from the plain version


def load_and_split(pdf_path: Path):
    """Load a PDF and split it into chunk Documents with clean metadata."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()  # one Document per page, metadata: {"source", "page" (0-based)}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(pages)

    # Normalize metadata to match the plain pipeline: filename + 1-based page
    ids = []
    counters: dict[int, int] = {}
    for doc in docs:
        page = int(doc.metadata.get("page", 0)) + 1
        chunk_no = counters.get(page, 0)
        counters[page] = chunk_no + 1
        doc.metadata = {"source": pdf_path.name, "page": page}
        ids.append(f"{pdf_path.stem}-p{page}-c{chunk_no}")
    return docs, ids


def main():
    if len(sys.argv) > 1:
        pdf_paths = [Path(sys.argv[1])]
    else:
        pdf_paths = sorted(Path("data").glob("*.pdf"))

    if not pdf_paths:
        print("No PDFs found. Drop .pdf files into ./data or pass a path.")
        sys.exit(1)

    pc, _ = get_clients()
    index = ensure_index(pc)

    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=PINECONE_API_KEY,
    )
    # text_key="text" stores each chunk's text in metadata["text"],
    # exactly like the plain pipeline does.
    store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

    for path in pdf_paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        print(f"\nProcessing {path.name}")
        docs, ids = load_and_split(path)
        if not docs:
            print("  No extractable text found (scanned PDF?). Skipping.")
            continue
        print(f"  {len(docs)} chunks")
        store.add_documents(docs, ids=ids)
        print(f"  Upserted {len(docs)} vectors.")

    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"\nDone. Index now holds {stats.get('total_vector_count', '?')} vectors.")


if __name__ == "__main__":
    main()
