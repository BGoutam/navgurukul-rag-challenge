"""
eval.py — Background LLM-as-judge scorer for every RAG response.

Pattern ported from Atticus / Bearpaw (`app/core/eval_service.py`).
Reimplemented from scratch with RAG-specific dimensions.

How it works:
    1. After /chat delivers a response, main.py calls `service.submit(EvalJob(...))`
    2. submit() is non-blocking — puts the job on an asyncio.Queue (max 100)
    3. A background coroutine consumes the queue, scores each response with
       Claude Haiku, and appends results to data/evals.jsonl

Five RAG-specific dimensions (each 1-5):
    relevance         — does the answer address what was asked?
    citation_accuracy — every [<pdf> p.<n>] maps to a real chunk + supports the claim?
    faithfulness      — every fact is in the chunks, no external-knowledge hallucination?
    completeness      — full coverage of the question, or partial?
    response_quality  — clear, well-formatted, professional, no padding?

Weighted overall → PASS (>=3.5) / PARTIAL (2.5-3.49) / FAIL (<2.5).

Results in data/evals.jsonl. The /eval-stats endpoint aggregates this file.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

EVAL_MODEL = "claude-haiku-4-5-20251001"
EVAL_HARD_TIMEOUT_S = 15.0

# Default location; main.py can override on init
DEFAULT_EVALS_PATH = Path("data/evals.jsonl")

WEIGHTS: Dict[str, float] = {
    "relevance":         0.25,
    "citation_accuracy": 0.25,
    "faithfulness":      0.25,
    "completeness":      0.15,
    "response_quality":  0.10,
}


@dataclass
class EvalJob:
    """Everything we need to score one /chat interaction."""
    question:  str
    answer:    str
    citations: List[Dict[str, Any]]
    is_idk:    bool
    chunks:    List[Dict[str, Any]]
    timing:    Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


def _build_eval_prompt(job: EvalJob) -> str:
    """The judge's prompt. The 5 chunk previews + answer + question."""
    chunks_str = "\n\n".join(
        f"[{c['pdf_name']} · p.{c['page']}]: {c.get('text_preview') or c.get('text', '')[:240]}"
        for c in job.chunks[:5]
    ) or "(no chunks retrieved)"

    return f"""You are evaluating one response from a Retrieval-Augmented Generation chatbot.

USER QUESTION:
{job.question}

RETRIEVED CHUNKS (top 5, what the assistant had access to):
{chunks_str}

ASSISTANT ANSWER:
{job.answer}

Score each dimension 1-5 (5 = excellent, 1 = poor). Return ONLY valid JSON, no prose:
{{
  "scores": {{
    "relevance":         <1-5>,
    "citation_accuracy": <1-5>,
    "faithfulness":      <1-5>,
    "completeness":      <1-5>,
    "response_quality":  <1-5>
  }},
  "strengths": ["<one-line strength>", ...],
  "issues":    ["<one-line issue>", ...]
}}

Definitions:
- relevance:         Does the answer address what was actually asked?
- citation_accuracy: Every [<pdf> p.<n>] in the answer matches a real chunk above AND the chunk supports that claim?
- faithfulness:      Every factual claim is present in the chunks. No external knowledge, no hallucination.
- completeness:      Full coverage of the question, or partial / superficial?
- response_quality:  Clear formatting, no padding, professional. (Soft signal; should not dominate.)

Special case: if the answer is EXACTLY "I don't have enough information in the indexed documents to answer this." — this is an honest refusal. Score relevance/citation/faithfulness 5 IF the chunks indeed do not contain the answer. Score completeness 3 (refused but didn't cover). Score response_quality 5."""


class EvalService:
    """Background eval queue. Construct once at app startup, call start() to begin."""

    def __init__(self, api_key: str, evals_path: Path = DEFAULT_EVALS_PATH):
        self.api_key = api_key
        self.evals_path = evals_path
        self.evals_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: asyncio.Queue[EvalJob] = asyncio.Queue(maxsize=100)
        self._task: Optional[asyncio.Task] = None
        self._stats = {"submitted": 0, "scored": 0, "errored": 0, "dropped": 0}

    # ── Public ──────────────────────────────────────────────────────────────

    def submit(self, job: EvalJob) -> None:
        """Fire-and-forget. Drops oldest if queue is full; never blocks caller."""
        self._stats["submitted"] += 1
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            logger.warning("[eval] queue full — dropping oldest")
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(job)
            except Exception:
                pass

    async def start(self) -> None:
        """Start the consumer loop. Idempotent."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
            logger.info("[eval] background service started")

    def stats(self) -> Dict[str, Any]:
        """Operational counters — exposed via /eval-stats."""
        return {**self._stats, "queue_size": self._queue.qsize()}

    # ── Internal loop ───────────────────────────────────────────────────────

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                result = await self._eval(job)
                self._save(job, result)
                self._stats["scored"] += 1
                logger.info(
                    f"[eval] verdict={result.get('verdict')} "
                    f"overall={result.get('overall')} "
                    f"latency={result.get('latency_ms')}ms"
                )
            except Exception as exc:
                self._stats["errored"] += 1
                logger.error(f"[eval] failed: {exc!r}")
            finally:
                self._queue.task_done()

    async def _eval(self, job: EvalJob) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "no api key", "verdict": "ERROR", "overall": 0.0}

        client = anthropic.AsyncAnthropic(api_key=self.api_key, timeout=EVAL_HARD_TIMEOUT_S)
        t0 = time.monotonic()
        resp = await client.messages.create(
            model=EVAL_MODEL,
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": _build_eval_prompt(job)}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            return {"error": f"non-json output: {exc!r}", "verdict": "ERROR",
                    "overall": 0.0, "latency_ms": latency_ms, "raw": raw[:200]}

        scores = parsed.get("scores", {})
        # Compute weighted overall (don't trust whatever overall the model emitted)
        overall = sum(float(scores.get(k, 3)) * w for k, w in WEIGHTS.items())
        verdict = "PASS" if overall >= 3.5 else "PARTIAL" if overall >= 2.5 else "FAIL"

        return {
            "scores":     scores,
            "overall":    round(overall, 2),
            "verdict":    verdict,
            "strengths":  parsed.get("strengths", []),
            "issues":     parsed.get("issues", []),
            "latency_ms": latency_ms,
        }

    def _save(self, job: EvalJob, result: Dict[str, Any]) -> None:
        entry = {
            "ts":             job.timestamp,
            "question":       job.question,
            "answer_preview": job.answer[:300],
            "is_idk":         job.is_idk,
            "n_chunks":       len(job.chunks),
            "n_citations":    len(job.citations),
            "timing":         job.timing,
            "eval":           result,
        }
        with open(self.evals_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Module-level singleton accessor (so main.py + tests share one instance) ──

_service: Optional[EvalService] = None


def init_service(api_key: str, evals_path: Path = DEFAULT_EVALS_PATH) -> EvalService:
    global _service
    if _service is None:
        _service = EvalService(api_key, evals_path)
    return _service


def get_service() -> Optional[EvalService]:
    return _service


# ── Aggregator (read-side, used by /eval-stats) ──────────────────────────────

def aggregate(evals_path: Path = DEFAULT_EVALS_PATH, last_n: int = 50) -> Dict[str, Any]:
    """
    Read the JSONL log, return per-dimension averages + verdict counts for the
    last N entries.
    """
    if not evals_path.exists():
        return {"n": 0, "by_dim": {}, "verdicts": {}, "recent": []}

    entries: List[Dict[str, Any]] = []
    with open(evals_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue

    recent = entries[-last_n:]
    if not recent:
        return {"n": 0, "by_dim": {}, "verdicts": {}, "recent": []}

    # Per-dimension averages
    dim_totals: Dict[str, List[float]] = {k: [] for k in WEIGHTS}
    for e in recent:
        scores = e.get("eval", {}).get("scores", {})
        for dim in WEIGHTS:
            v = scores.get(dim)
            if v is not None:
                dim_totals[dim].append(float(v))
    by_dim = {
        k: {
            "avg":   round(sum(v) / len(v), 2) if v else None,
            "count": len(v),
        }
        for k, v in dim_totals.items()
    }

    # Verdict counts
    verdicts: Dict[str, int] = {}
    overall_total = 0.0
    overall_count = 0
    for e in recent:
        ev = e.get("eval", {})
        v = ev.get("verdict", "UNKNOWN")
        verdicts[v] = verdicts.get(v, 0) + 1
        o = ev.get("overall")
        if isinstance(o, (int, float)):
            overall_total += float(o)
            overall_count += 1

    avg_overall = round(overall_total / overall_count, 2) if overall_count else None

    # Compact recent summary for the UI
    recent_summary = [
        {
            "ts":            e.get("ts"),
            "question":      (e.get("question") or "")[:120],
            "verdict":       e.get("eval", {}).get("verdict"),
            "overall":       e.get("eval", {}).get("overall"),
            "is_idk":        e.get("is_idk"),
            "issues":        e.get("eval", {}).get("issues", [])[:2],
        }
        for e in reversed(recent[-10:])  # newest first, last 10 of the window
    ]

    return {
        "n":           len(recent),
        "avg_overall": avg_overall,
        "by_dim":      by_dim,
        "verdicts":    verdicts,
        "recent":      recent_summary,
    }
