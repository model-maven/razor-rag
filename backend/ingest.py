"""
Ingest PDFs into Pinecone.

Usage:
    python ingest.py                  # ingests every PDF in ./data
    python ingest.py path/to/file.pdf # ingests a single PDF
"""
import sys
import time
from pathlib import Path

from pypdf import PdfReader
from pinecone import ServerlessSpec

from config import (
    get_clients,
    INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

EMBED_BATCH = 90   # Pinecone inference accepts up to ~96 inputs per call
UPSERT_BATCH = 100


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return a list of {page_number, text} for a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, trying to break on sentence/word boundaries."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # try to end at a sentence boundary, then a space
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut < size // 2:
                cut = window.rfind(" ")
            if cut > size // 2:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def build_chunks(pdf_path: Path) -> list[dict]:
    """Extract and chunk one PDF into records ready for embedding."""
    records = []
    for page in extract_pages(pdf_path):
        for j, chunk in enumerate(chunk_text(page["text"])):
            records.append({
                "id": f"{pdf_path.stem}-p{page['page']}-c{j}",
                "text": chunk,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page["page"],
                    "text": chunk,
                },
            })
    return records


def ensure_index(pc):
    existing = [ix["name"] for ix in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # wait until ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(INDEX_NAME)


def embed_texts(pc, texts: list[str]) -> list[list[float]]:
    """Embed passages with Pinecone's hosted embedding model."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        result = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        vectors.extend([item["values"] for item in result])
        print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)} chunks")
    return vectors


def ingest_pdf(pc, index, pdf_path: Path):
    print(f"\nProcessing {pdf_path.name}")
    records = build_chunks(pdf_path)
    if not records:
        print("  No extractable text found (is this a scanned PDF?). Skipping.")
        return
    print(f"  {len(records)} chunks")

    vectors = embed_texts(pc, [r["text"] for r in records])

    to_upsert = [
        {"id": r["id"], "values": v, "metadata": r["metadata"]}
        for r, v in zip(records, vectors)
    ]
    for i in range(0, len(to_upsert), UPSERT_BATCH):
        index.upsert(vectors=to_upsert[i:i + UPSERT_BATCH])
    print(f"  Upserted {len(to_upsert)} vectors.")


def main():
    if len(sys.argv) > 1:
        pdf_paths = [Path(sys.argv[1])]
    else:
        pdf_paths = sorted(Path("data").glob("*.pdf"))

    if not pdf_paths:
        print("No PDFs found. Drop .pdf files into the ./data folder or pass a path:")
        print("  python ingest.py path/to/file.pdf")
        sys.exit(1)

    pc, _ = get_clients()
    index = ensure_index(pc)

    for path in pdf_paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        ingest_pdf(pc, index, path)

    time.sleep(2)  # let stats settle
    stats = index.describe_index_stats()
    print(f"\nDone. Index now holds {stats.get('total_vector_count', '?')} vectors.")


if __name__ == "__main__":
    main()
