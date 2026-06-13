# Navgurukul RAG Challenge — Submission

> A Retrieval-Augmented Generation chatbot answering questions from a corpus
> of 10 PDFs (2,088 pages of NLP / ML literature). Open-source retrieval
> stack, cited answers, 2–5s end-to-end latency.

## Status

**Phase 1 complete:** PDF ingestion pipeline (text + OCR fallback) → deterministic
chunking → Sentence Transformers embedding → ChromaDB. CLI works end-to-end.

**Phases pending:** retrieve, generate, web UI, input guardrail, auto-eval,
audit chain, benchmark metrics. See `../Bearpaw/Navgurukul_plan.md` for the
full plan and acceptance criteria per phase.

## Quick install

```bash
# From the repo root
pip install -r requirements.txt

# OCR — Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# OCR — macOS:   brew install tesseract
# OCR — Linux:   sudo apt-get install tesseract-ocr
```

Then populate `data/pdfs/` (see `data/pdfs/README.md`) or point the ingest
script at any directory containing PDFs.

## Phase 1: Ingest the corpus

```bash
# Ingest the augmented Transformers corpus directly from Bearpaw
python scripts/ingest_pdfs.py ../Bearpaw/Transformers/
```

What this does:
1. Walks the directory recursively, finds every `.pdf`
2. For each PDF, extracts text page-by-page with `pdfplumber`. If a page has
   <50 words of native text, falls back to Tesseract OCR
3. Splits into ~700-token chunks with 15% overlap (deterministic — same
   input always produces the same chunks → reproducible embeddings)
4. Embeds with `sentence-transformers/all-MiniLM-L6-v2` (open-source, 384-dim)
5. Writes to ChromaDB at `data/chroma_db/` with metadata `{pdf_name, page, chunk_index}`

Idempotency: a per-PDF SHA-256 is recorded in `data/chroma_db/manifest.json`.
Re-running skips unchanged PDFs unless `--force` is passed.

## Stack — open-source compliance

| Layer | Choice | License |
|-------|--------|---------|
| PDF text | `pdfplumber` + `pypdfium2` | MIT |
| OCR | `pytesseract` + Tesseract | Apache 2.0 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 |
| Vector DB | ChromaDB | Apache 2.0 |
| LLM (planned) | Anthropic Claude (hosted; spec allows) | Hosted |
| Web (planned) | FastAPI + uvicorn | MIT/BSD |

Every retrieval-side component (embedding, vector DB, OCR, PDF text) is
free and open-source. Generation is hosted Claude — allowed by the spec
("open-source OR hosted").

## Patterns inherited from Atticus / Bearpaw

This submission reimplements three patterns from a larger agent operating
system called Atticus (`../Bearpaw/`):

- **Input guardrail** — Claude Haiku safety classifier that blocks prompt
  injection, PII leaks, and off-policy use before any retrieval happens.
- **LLM-as-judge auto-eval** — every response is scored on five RAG-specific
  dimensions (relevance, citation accuracy, faithfulness, completeness,
  response quality) in a background queue.
- **Merkle audit chain** — hash-linked append-only JSONL log of every
  interaction; tamper-evident.

The implementations in this repo are clean rewrites of those patterns, not
imports. The point of the submission is the focused slice: a 1300-LOC RAG
that's measurable, auditable, and cites every answer.
