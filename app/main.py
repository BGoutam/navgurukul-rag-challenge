"""
main.py — FastAPI gateway. Serves the chat UI and exposes the RAG endpoints.

Routes:
    GET  /                 chat UI (static/index.html)
    POST /chat             question → cited answer
    GET  /ingest-status    ChromaDB population stats
    GET  /health           liveness probe
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env BEFORE we import anything that may read env vars
from dotenv import load_dotenv

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
load_dotenv(PROJECT_ROOT / ".env")

import asyncio
import json as _json
import queue as _stdq
import threading
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Local
import chromadb
from chromadb.config import Settings as ChromaSettings

from .ingest import COLLECTION_NAME, ingest_directory  # noqa: E402
from .retrieve import retrieve  # noqa: E402
from .generate import generate  # noqa: E402
from .guardrail import check_input, GuardrailVerdict  # noqa: E402
from .eval import (  # noqa: E402
    EvalJob, init_service as init_eval_service,
    get_service as get_eval_service, aggregate as aggregate_evals,
    DEFAULT_EVALS_PATH,
)
from .audit import init_chain as init_audit, log_action as audit_log, get_chain as get_audit  # noqa: E402

# Silence ChromaDB telemetry (same pattern as ingest.py)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)

# Filesystem layout
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
STATIC_DIR = PROJECT_ROOT / "static"
EVALS_PATH = PROJECT_ROOT / "data" / "evals.jsonl"
AUDIT_PATH = PROJECT_ROOT / "data" / "audit.jsonl"

app = FastAPI(title="Navgurukul RAG Chatbot", version="0.4.0")


# ── Startup: init audit chain + kick off the background eval queue ──────────

@app.on_event("startup")
async def _startup():
    init_audit(AUDIT_PATH)
    logger.info(f"Audit chain initialised at {AUDIT_PATH}")

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        svc = init_eval_service(api_key, EVALS_PATH)
        await svc.start()
        logger.info("Eval background service started.")
    else:
        logger.warning("ANTHROPIC_API_KEY not set — eval service inactive.")

# Permissive CORS — local-only demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(8, ge=1, le=25, description="Top-K chunks to retrieve.")
    rerank: bool = Field(False, description="Apply CrossEncoder reranking.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    """Serve the single-page chat UI."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "static/index.html missing")
    return FileResponse(index)


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Pipeline (per spec): input guardrail → retrieve → generate → background eval.
    On guardrail block: return early with the block message, never call retrieve/generate.
    On guardrail pass: full retrieve + generate, then enqueue background eval.
    """
    if not CHROMA_DB_PATH.exists():
        raise HTTPException(
            503,
            "ChromaDB not initialised. Run `python scripts/ingest_pdfs.py <pdf_dir>` first.",
        )

    # ── Audit: user input received ──────────────────────────────────────────
    audit_log("USER_INPUT", {"question": req.question[:500], "k": req.k, "rerank": req.rerank})

    # ── 1. Input guardrail (sequential; ~1.3s typical) ──────────────────────
    guard = await check_input(req.question)
    audit_log("GUARDRAIL_CHECK", guard.to_dict())
    if not guard.passed:
        block_msg = (
            f"Your request was blocked by the input guardrail "
            f"(`{guard.verdict.value}`, confidence {guard.confidence:.2f}).\n\n"
            f"**Reason:** {guard.reason}"
        )
        return {
            "question":  req.question,
            "answer":    block_msg,
            "blocked":   True,
            "guardrail": guard.to_dict(),
            "citations": [],
            "is_idk":    False,
            "chunks":    [],
            "timing": {
                "guardrail_ms":  guard.latency_ms,
                "retrieval_ms":  0,
                "rerank_ms":     None,
                "generation_ms": 0,
                "total_ms":      guard.latency_ms,
            },
            "usage": {"input_tokens": 0, "output_tokens": 0, "model": "blocked"},
        }

    # ── 2. Retrieve ─────────────────────────────────────────────────────────
    retrieval = retrieve(
        query=req.question,
        chroma_db_path=CHROMA_DB_PATH,
        k=req.k,
        rerank=req.rerank,
    )

    # ── 3. Generate ─────────────────────────────────────────────────────────
    answer = generate(req.question, retrieval["chunks"])

    total_ms = (
        guard.latency_ms
        + retrieval["retrieval_ms"]
        + (retrieval["rerank_ms"] or 0)
        + answer["generation_ms"]
    )

    response_chunks = [
        {
            "pdf_name":     c["pdf_name"],
            "page":         c["page"],
            "score":        c["score"],
            "rerank_score": c.get("rerank_score"),
            "text_preview": c["text"][:240],
        }
        for c in retrieval["chunks"]
    ]

    response = {
        "question":  req.question,
        "answer":    answer["answer"],
        "blocked":   False,
        "guardrail": guard.to_dict(),
        "citations": answer["citations"],
        "is_idk":    answer["is_idk"],
        "chunks":    response_chunks,
        "timing": {
            "guardrail_ms":  guard.latency_ms,
            "retrieval_ms":  retrieval["retrieval_ms"],
            "rerank_ms":     retrieval["rerank_ms"],
            "generation_ms": answer["generation_ms"],
            "total_ms":      total_ms,
        },
        "usage": {
            "input_tokens":  answer["input_tokens"],
            "output_tokens": answer["output_tokens"],
            "model":         answer["model"],
        },
    }

    # ── 4. Audit: full chat record ──────────────────────────────────────────
    audit_log("CHAT_COMPLETE", {
        "question_preview": req.question[:200],
        "answer_preview":   answer["answer"][:300],
        "is_idk":           answer["is_idk"],
        "n_chunks":         len(response_chunks),
        "n_citations":      len(answer["citations"]),
        "timing":           response["timing"],
        "input_tokens":     answer["input_tokens"],
        "output_tokens":    answer["output_tokens"],
    })

    # ── 5. Fire-and-forget background eval (never blocks the user) ──────────
    svc = get_eval_service()
    if svc is not None:
        svc.submit(EvalJob(
            question=req.question,
            answer=answer["answer"],
            citations=answer["citations"],
            is_idk=answer["is_idk"],
            chunks=response_chunks,
            timing=response["timing"],
        ))

    return response


@app.get("/eval-stats")
async def eval_stats(last_n: int = 50):
    """Aggregated quality scores across the last N evaluated responses."""
    last_n = max(1, min(int(last_n), 500))
    agg = aggregate_evals(EVALS_PATH, last_n=last_n)
    svc = get_eval_service()
    agg["queue"] = svc.stats() if svc else {"submitted": 0, "scored": 0, "errored": 0,
                                            "dropped": 0, "queue_size": 0}
    return agg


@app.get("/audit/log")
async def audit_log_endpoint(limit: int = Query(50, ge=1, le=500)):
    """Return the most recent N audit-chain entries (newest first) + chain verify."""
    chain = get_audit()
    if chain is None:
        raise HTTPException(503, "Audit chain not initialised.")
    return {
        "verify":  chain.verify(),
        "entries": chain.recent(limit),
    }


# ── Ingestion via SSE — streams progress per-page to the UI ──────────────────

async def _ingest_sse(pdf_dir: Path, force: bool) -> AsyncGenerator[str, None]:
    """
    Run ingest_directory in a worker thread, surface progress events to the
    browser as Server-Sent Events.
    """
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        yield f"event: error\ndata: {_json.dumps({'error': f'not a directory: {pdf_dir}'})}\n\n"
        return

    audit_log("INGEST_START", {"pdf_dir": str(pdf_dir), "force": force})

    q: _stdq.Queue = _stdq.Queue()
    state: dict = {"done": False, "stats": None, "error": None}

    def progress(pdf_name, page=None, total=None, status=None, reason=None):
        q.put({
            "type":     "progress",
            "pdf_name": pdf_name,
            "page":     page,
            "total":    total,
            "status":   status,
            "reason":   reason,
        })

    def run_in_thread():
        try:
            stats = ingest_directory(pdf_dir, CHROMA_DB_PATH, force=force, progress=progress)
            state["stats"] = stats
        except Exception as exc:
            state["error"] = repr(exc)
            logger.error(f"[ingest/sse] failed: {exc!r}")
        finally:
            state["done"] = True

    t = threading.Thread(target=run_in_thread, daemon=True)
    t.start()

    # First event so the browser knows the stream is live
    yield f"data: {_json.dumps({'type': 'start', 'pdf_dir': str(pdf_dir)})}\n\n"

    # Drain progress queue until the worker thread finishes
    while True:
        try:
            evt = q.get(timeout=0.25)
            yield f"data: {_json.dumps(evt)}\n\n"
        except _stdq.Empty:
            if state["done"]:
                break
            # Heartbeat comment so proxies don't drop the connection on idle
            yield ": keepalive\n\n"
            await asyncio.sleep(0)

    if state["error"]:
        audit_log("INGEST_ERROR", {"error": state["error"]})
        yield f"data: {_json.dumps({'type': 'error', 'error': state['error']})}\n\n"
    else:
        audit_log("INGEST_COMPLETE", {
            "pdfs_total":     state["stats"]["pdfs_total"],
            "pdfs_ingested":  state["stats"]["pdfs_ingested"],
            "pdfs_skipped":   state["stats"]["pdfs_skipped"],
            "total_pages":    state["stats"]["total_pages"],
            "total_chunks":   state["stats"]["total_chunks"],
        })
        yield f"data: {_json.dumps({'type': 'done', 'stats': state['stats']})}\n\n"


@app.get("/ingest/run")
async def ingest_run(
    dir: str = Query(..., description="PDF directory to ingest (relative or absolute)."),
    force: bool = Query(False, description="Re-ingest even if SHA-256 matches manifest."),
):
    """Run ingestion with live SSE progress. Browser opens via EventSource."""
    pdf_dir = Path(dir)
    if not pdf_dir.is_absolute():
        pdf_dir = (PROJECT_ROOT / pdf_dir).resolve()
    return StreamingResponse(
        _ingest_sse(pdf_dir, force),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",  # for any reverse proxy in the way
        },
    )


@app.get("/ingest-status")
async def ingest_status():
    """Current ChromaDB stats — used by the UI to show 'ready'/'empty' badges."""
    if not CHROMA_DB_PATH.exists():
        return {
            "populated":   False,
            "vector_count": 0,
            "pdfs":        [],
            "n_pdfs":      0,
        }

    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        count = collection.count()
    except Exception as exc:
        logger.warning(f"ingest_status: cannot read collection: {exc}")
        count = 0

    # Read manifest for the PDF list
    pdfs: List[str] = []
    manifest_path = CHROMA_DB_PATH / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pdfs = sorted(manifest.keys())
        except Exception:
            pass

    return {
        "populated":    count > 0,
        "vector_count": count,
        "n_pdfs":       len(pdfs),
        "pdfs":         pdfs,
    }


# Static mount LAST so it doesn't shadow named routes.
# This serves any future assets from /static/<file>.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
