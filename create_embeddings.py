"""Build the FAISS vector store from PDFs in the papers directory.

This script preserves source filename and page metadata so the downstream RAG
pipeline can cite papers as `[Paper: file.pdf, Page N]`.
"""

import pickle
from typing import List

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    PAPERS_DIR,
    VECTOR_STORE_DIR,
)


def load_pdf_pages() -> List[Document]:
    """Load all PDF pages and normalize metadata for later citation."""

    documents: List[Document] = []
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PAPERS_DIR}")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["paper"] = pdf_path.name
                # PyPDFLoader pages are zero-indexed; citations should be human
                # readable and one-indexed.
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1

            documents.extend(pages)
            print(f"Loaded: {pdf_path.name} ({len(pages)} pages)")
        except Exception as exc:
            print(f"Error loading {pdf_path.name}: {exc}")

    print(f"Total pages loaded: {len(documents)}")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks while retaining metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata.setdefault("paper", chunk.metadata.get("source", "unknown"))
        chunk.metadata.setdefault("page", "unknown")

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def build_faiss_index(chunks: List[Document]) -> faiss.Index:
    """Embed chunks and create a cosine-similarity FAISS index."""

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk.page_content for chunk in chunks]
    print("Generating normalized embeddings...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors")
    return index


def save_vector_store(index: faiss.Index, chunks: List[Document]) -> None:
    """Persist the FAISS index and chunk metadata."""

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    print(f"Saved chunks to {CHUNKS_PATH}")


def main() -> None:
    documents = load_pdf_pages()
    chunks = split_documents(documents)
    index = build_faiss_index(chunks)
    save_vector_store(index, chunks)
    print("Embedding creation completed successfully")


if __name__ == "__main__":
    main()

