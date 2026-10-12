"""PDF loading, page metadata, chunking, and FAISS indexing."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


def _embeddings():
    if not API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY not found. Add it to your .env file.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)


def load_pdf(path: str, source_name: str | None = None):
    """Load text-based PDF pages and attach stable, user-facing source metadata."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    docs = PyPDFLoader(str(file_path)).load()
    label = source_name or file_path.name
    for doc in docs:
        doc.metadata["source_file"] = label
        doc.metadata["page_display"] = int(doc.metadata.get("page", 0)) + 1
    return docs


def split_documents(docs):
    """Split pages independently while preserving their source/page metadata."""
    nonempty = [doc for doc in docs if getattr(doc, "page_content", "").strip()]
    if not nonempty:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(nonempty)


def build_vector_db(chunks):
    if not chunks:
        raise ValueError("Cannot build a vector index without text chunks.")
    return FAISS.from_documents(chunks, _embeddings())


def add_documents(db, chunks):
    if not chunks:
        raise ValueError("No text chunks were provided.")
    if db is None:
        return build_vector_db(chunks)
    new_db = FAISS.from_documents(chunks, _embeddings())
    db.merge_from(new_db)
    return db


def chunk_stats(chunks):
    sizes = [len(chunk.page_content) for chunk in chunks]
    return {
        "count": len(chunks),
        "avg_size": round(sum(sizes) / len(sizes)) if sizes else 0,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
    }
