"""
generate.py — Given retrieved chunks + a query, generate a cited answer.

Hard-instructed citation discipline: every factual claim must end with
[<filename> p.<n>]. If the chunks don't contain the answer, the model
returns a fixed I-don't-know string instead of hallucinating.

Uses Claude Sonnet 4.6 (hosted; spec allows). Temperature 0.0 for
factuality. The hard citation rule and the fixed IDK string are the two
levers that drive citation accuracy and hallucination rate in eval.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List

import anthropic

logger = logging.getLogger(__name__)

GENERATION_MODEL = "claude-sonnet-4-6"
GENERATION_MAX_TOKENS = 1024
GENERATION_TEMPERATURE = 0.0   # factuality > creativity for RAG

IDK_RESPONSE = "I don't have enough information in the indexed documents to answer this."

_SYSTEM_PROMPT = """You are a Retrieval-Augmented Generation assistant. Answer ONLY from the chunks provided. Follow these rules exactly:

1. CITATIONS: Every factual claim must end with a citation in the form [<filename> p.<n>]. Use the exact filename and page number from the chunk's header.

2. NO HALLUCINATION: If the chunks don't contain enough information to answer the question, reply with this exact string and nothing else:
   "I don't have enough information in the indexed documents to answer this."
   Do not invent citations. Do not use external knowledge.

3. CONCISE: Answer directly. Don't restate the question. Don't pad with summary. One citation per claim — not three for the same fact.

4. NO META: Don't say "Based on the chunks..." or "According to the documents...". Just answer.

5. STRUCTURE: For multi-part questions, use brief markdown — bullet points or numbered lists. Inline citations stay at end of each claim."""


def _build_user_message(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Format the retrieved chunks + question for the LLM."""
    chunk_blocks = []
    for i, ck in enumerate(chunks, 1):
        chunk_blocks.append(
            f"[Chunk {i}] (from {ck['pdf_name']}, page {ck['page']})\n"
            f"{ck['text']}"
        )
    chunks_section = "\n\n".join(chunk_blocks)
    return (
        f"CHUNKS:\n{chunks_section}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"Now answer, citing chunks by [<filename> p.<n>] for every claim."
    )


def _extract_citations(answer: str) -> List[Dict[str, Any]]:
    """
    Pull out all [<filename> p.<n>] citations from the answer text.
    Returns a deduplicated list of {pdf_name, page} dicts in order of first appearance.
    """
    pattern = re.compile(r"\[([^\]]+?)\s+p\.\s*(\d+)\]")
    seen = set()
    out: List[Dict[str, Any]] = []
    for m in pattern.finditer(answer):
        pdf_name = m.group(1).strip()
        page = int(m.group(2))
        key = (pdf_name.lower(), page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"pdf_name": pdf_name, "page": page})
    return out


def generate(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a cited answer from the retrieved chunks.

    Returns:
        {
          "answer":         str,
          "citations":      [{pdf_name, page}, ...],
          "is_idk":         bool,   # True if model returned the fixed IDK string
          "generation_ms":  int,
          "input_tokens":   int,
          "output_tokens":  int,
          "model":          str,
        }
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return {
            "answer":        "Error: ANTHROPIC_API_KEY is not set in the environment.",
            "citations":     [],
            "is_idk":        False,
            "generation_ms": 0,
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         GENERATION_MODEL,
            "error":         "missing_api_key",
        }

    if not chunks:
        return {
            "answer":        IDK_RESPONSE,
            "citations":     [],
            "is_idk":        True,
            "generation_ms": 0,
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         GENERATION_MODEL,
        }

    client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
    user_msg = _build_user_message(query, chunks)

    t0 = time.monotonic()
    try:
        resp = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        logger.error(f"Claude generation failed: {exc}")
        return {
            "answer":        f"Generation error: {exc}",
            "citations":     [],
            "is_idk":        False,
            "generation_ms": int((time.monotonic() - t0) * 1000),
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         GENERATION_MODEL,
            "error":         repr(exc),
        }
    gen_ms = int((time.monotonic() - t0) * 1000)

    answer = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    is_idk = answer.strip() == IDK_RESPONSE
    citations = _extract_citations(answer)

    return {
        "answer":        answer,
        "citations":     citations,
        "is_idk":        is_idk,
        "generation_ms": gen_ms,
        "input_tokens":  resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "model":         GENERATION_MODEL,
    }
