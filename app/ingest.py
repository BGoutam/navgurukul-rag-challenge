"""
ingest.py — PDF ingestion pipeline for the RAG corpus.

Pipeline per PDF:
    1. Native text extraction via pdfplumber
    2. OCR fallback via pypdfium2 + Tesseract for sparse pages (e.g. scanned)
    3. Deterministic chunking — fixed token target with fixed overlap
    4. Sentence Transformers embedding (open-source, local)
    5. Persist to ChromaDB collection with metadata {pdf_name, page, chunk_index}

Idempotency: per-PDF SHA-256 is recorded in a manifest. Re-running skips
unchanged PDFs unless `force=True` is passed.

This module is the ENTIRE retrieval-side stack. No paid services, no cloud
embeddings — verifiable open-source compliance for the challenge.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

# Silence ChromaDB's posthog telemetry — the SDK ships a buggy posthog version
# that emits noisy "capture() takes 1 positional argument but 3 were given"
# errors and occasionally blocks on the telemetry HTTP call. Free to disable.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Target chunk size in approximate tokens. 700 is a good middle ground — large
# enough to carry context, small enough that the LLM can attend to all retrieved
# chunks in a single generation call.
CHUNK_SIZE_TOKENS = 700

# Overlap ratio between adjacent chunks (15%). Helps preserve context around
# sentences that would otherwise be split.
CHUNK_OVERLAP_RATIO = 0.15

# If native PDF text extraction returns fewer than this many words for a page,
# we treat it as scanned / image-heavy and run OCR instead.
OCR_FALLBACK_WORD_THRESHOLD = 50

# Sentence Transformers model. all-MiniLM-L6-v2 is fast (384-dim) and well
# tested for retrieval; BAAI/bge-small-en-v1.5 is a quality upgrade if needed.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Single ChromaDB collection holds the entire corpus.
COLLECTION_NAME = "navgurukul_corpus"

# Tesseract language pack.
TESSERACT_LANG = "eng"

# Render scale for OCR — 2x means 200 DPI from PDF's nominal 72 DPI.
OCR_RENDER_SCALE = 2

# Add chunks to ChromaDB in batches of this size — single-shot inserts of
# thousands of chunks slow Chroma's index updates.
CHROMA_BATCH_SIZE = 200


# ── Tesseract path resolution (Windows often needs an explicit path) ─────────

_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
elif os.name == "nt":
    # Best-effort default for Windows installs of Tesseract
    _default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(_default).exists():
        pytesseract.pytesseract.tesseract_cmd = _default


# ── Utilities ────────────────────────────────────────────────────────────────

def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Stream-hash a file. Used to detect content changes for idempotent reruns."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _approx_words_per_chunk() -> int:
    """Approximate word count for our target token budget. ~1.3 tokens/word."""
    return int(CHUNK_SIZE_TOKENS / 1.3)


# ── Page extraction with OCR fallback ─────────────────────────────────────────

def extract_page_text(pdf_path: Path, page_num: int) -> str:
    """
    Extract text from one page (0-indexed).

    Tries native pdfplumber extraction first. If the result has fewer than
    OCR_FALLBACK_WORD_THRESHOLD words, falls back to OCR via Tesseract.
    """
    # Native extraction first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return ""
            page = pdf.pages[page_num]
            native_text = page.extract_text() or ""
            if len(native_text.split()) >= OCR_FALLBACK_WORD_THRESHOLD:
                return native_text
    except Exception as exc:
        logger.warning(f"Native extract failed for {pdf_path.name} p{page_num+1}: {exc}")

    # OCR fallback
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        if page_num >= len(doc):
            return ""
        page = doc[page_num]
        pil_image = page.render(scale=OCR_RENDER_SCALE).to_pil()
        return pytesseract.image_to_string(pil_image, lang=TESSERACT_LANG)
    except Exception as exc:
        logger.error(f"OCR fallback failed for {pdf_path.name} p{page_num+1}: {exc}")
        return ""


# ── Chunker — deterministic, no randomness ────────────────────────────────────

def chunk_text(
    text: str,
    page_num: int,
    pdf_name: str,
) -> List[Dict[str, Any]]:
    """
    Split a page's text into fixed-size chunks with overlap.

    Returns a list of {'text': str, 'metadata': dict} entries.
    Deterministic: same input → same output → reproducible embeddings.
    """
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []

    words_per_chunk = _approx_words_per_chunk()
    overlap_words = int(words_per_chunk * CHUNK_OVERLAP_RATIO)
    step = max(1, words_per_chunk - overlap_words)

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append({
            "text": chunk_str,
            "metadata": {
                "pdf_name": pdf_name,
                "page": page_num + 1,  # 1-indexed in metadata — what we cite to the user
                "chunk_index": chunk_index,
                "char_count": len(chunk_str),
                "word_count": len(chunk_words),
            },
        })
        chunk_index += 1
        if end >= len(words):
            break
        start += step
    return chunks


# ── Per-PDF ingestion ────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    collection,
    manifest: Dict[str, str],
    force: bool = False,
    progress: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """
    Ingest one PDF. Returns stats dict.

    If the file's SHA-256 matches the manifest entry and force=False, skips.
    Otherwise extracts every page, chunks, embeds, and writes to ChromaDB.
    """
    pdf_name = pdf_path.name
    sha = file_sha256(pdf_path)

    if not force and manifest.get(pdf_name) == sha:
        if progress:
            progress(pdf_name, status="SKIP", reason="unchanged")
        return {
            "pdf_name": pdf_name,
            "sha256": sha,
            "action": "skipped",
            "n_pages": 0,
            "n_chunks": 0,
        }

    # If we're re-ingesting, drop any existing chunks for this PDF so we don't
    # accumulate duplicates. Chroma's delete-by-where is the cleanest path.
    try:
        collection.delete(where={"pdf_name": pdf_name})
    except Exception as exc:  # collection might be empty — that's fine
        logger.debug(f"Delete pre-ingest for {pdf_name}: {exc}")

    # Page count
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
    except Exception as exc:
        logger.error(f"Cannot open {pdf_name}: {exc}")
        return {"pdf_name": pdf_name, "sha256": sha, "action": "error",
                "n_pages": 0, "n_chunks": 0, "error": str(exc)}

    # Extract + chunk page by page
    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    for page_num in range(n_pages):
        if progress:
            progress(pdf_name, page=page_num + 1, total=n_pages)
        text = extract_page_text(pdf_path, page_num)
        if not text.strip():
            continue
        for ck in chunk_text(text, page_num, pdf_name):
            chunk_id = f"{pdf_name}::{page_num+1}::{ck['metadata']['chunk_index']}"
            all_texts.append(ck["text"])
            all_ids.append(chunk_id)
            all_metas.append(ck["metadata"])

    # Batch-add to ChromaDB
    for i in range(0, len(all_texts), CHROMA_BATCH_SIZE):
        collection.add(
            documents=all_texts[i:i + CHROMA_BATCH_SIZE],
            ids=all_ids[i:i + CHROMA_BATCH_SIZE],
            metadatas=all_metas[i:i + CHROMA_BATCH_SIZE],
        )

    manifest[pdf_name] = sha

    return {
        "pdf_name": pdf_name,
        "sha256": sha,
        "action": "ingested",
        "n_pages": n_pages,
        "n_chunks": len(all_texts),
    }


# ── Top-level: walk a directory ──────────────────────────────────────────────

def ingest_directory(
    pdf_dir: Path,
    chroma_db_path: Path,
    force: bool = False,
    progress: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """
    Walk pdf_dir recursively and ingest every .pdf into the ChromaDB collection
    at chroma_db_path. Returns aggregate statistics.
    """
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    # ChromaDB client + collection (telemetry disabled — see top-of-file env vars)
    client = chromadb.PersistentClient(
        path=str(chroma_db_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # First call to this function downloads the embedding model from HuggingFace
    # (~90 MB) into ~/.cache/huggingface/ on first run. Can take 30-60s the
    # first time; instant on subsequent runs.
    if progress:
        progress("__init__", status="LOAD_MODEL", reason=EMBEDDING_MODEL)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Load manifest (which PDFs we've already ingested + their SHA-256)
    manifest_path = chroma_db_path / "manifest.json"
    manifest: Dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    pdfs = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])

    stats: Dict[str, Any] = {
        "pdfs_total": len(pdfs),
        "pdfs_ingested": 0,
        "pdfs_skipped": 0,
        "pdfs_error": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "details": [],
    }

    for pdf_path in pdfs:
        result = ingest_pdf(pdf_path, collection, manifest, force=force, progress=progress)
        stats["details"].append(result)
        if result["action"] == "ingested":
            stats["pdfs_ingested"] += 1
        elif result["action"] == "skipped":
            stats["pdfs_skipped"] += 1
        else:
            stats["pdfs_error"] += 1
        stats["total_pages"] += result.get("n_pages", 0)
        stats["total_chunks"] += result.get("n_chunks", 0)

    # Persist manifest
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Final collection size
    try:
        stats["collection_count"] = collection.count()
    except Exception:
        stats["collection_count"] = -1

    return stats
