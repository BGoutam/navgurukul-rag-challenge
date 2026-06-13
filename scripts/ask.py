#!/usr/bin/env python3
"""
ask.py — CLI: ask a question against the ingested RAG corpus.

Usage:
    python scripts/ask.py "What is multi-head attention?"
    python scripts/ask.py "Explain Q, K, V" --k 8
    python scripts/ask.py "How does positional encoding work?" --rerank
    python scripts/ask.py "..." --json     # machine-readable output

Reads the ChromaDB at data/chroma_db/ by default.
Requires ANTHROPIC_API_KEY in .env (or environment).
"""
import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_HERE))

try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
except ImportError:
    pass  # dotenv is optional; env vars work either way

from app.retrieve import retrieve  # noqa: E402
from app.generate import generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ask a question against the RAG corpus.")
    parser.add_argument("question", type=str, help="Question to ask (in quotes).")
    parser.add_argument("--db", type=Path, default=_HERE / "data" / "chroma_db",
                        help="ChromaDB path (default: data/chroma_db/).")
    parser.add_argument("--k", type=int, default=8,
                        help="Top-K chunks to retrieve (default: 8).")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply CrossEncoder reranking after retrieval (slower, better recall).")
    parser.add_argument("--rerank-top", type=int, default=5,
                        help="After rerank, keep top-N (default: 5).")
    parser.add_argument("--json", dest="emit_json", action="store_true",
                        help="Print raw JSON output instead of formatted text.")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Print the retrieved chunks before the answer.")
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"Error: ChromaDB not found at {args.db}. Run ingest_pdfs.py first.")

    # ── Retrieve ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    retrieval = retrieve(
        query=args.question,
        chroma_db_path=args.db,
        k=args.k,
        rerank=args.rerank,
        rerank_top_n=args.rerank_top,
    )
    chunks = retrieval["chunks"]

    # ── Generate ────────────────────────────────────────────────────────────
    answer = generate(args.question, chunks)
    total_ms = int((time.monotonic() - t0) * 1000)

    if args.emit_json:
        out = {
            "question":      args.question,
            "answer":        answer["answer"],
            "citations":     answer["citations"],
            "is_idk":        answer["is_idk"],
            "retrieval_ms":  retrieval["retrieval_ms"],
            "rerank_ms":     retrieval["rerank_ms"],
            "generation_ms": answer["generation_ms"],
            "total_ms":      total_ms,
            "input_tokens":  answer["input_tokens"],
            "output_tokens": answer["output_tokens"],
            "n_chunks":      len(chunks),
            "model":         answer["model"],
        }
        if args.show_chunks:
            out["chunks"] = chunks
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    # ── Human-readable output ───────────────────────────────────────────────
    print()
    print("─" * 70)
    print(f"  QUESTION: {args.question}")
    print("─" * 70)

    if args.show_chunks:
        print()
        print(f"  Retrieved {len(chunks)} chunks "
              f"({'reranked top ' + str(args.rerank_top) if args.rerank else 'top-K'}):")
        for i, ck in enumerate(chunks, 1):
            score_label = f"rerank={ck.get('rerank_score'):.3f} · " if args.rerank else ""
            print(f"    {i}. [{ck['pdf_name']} · p.{ck['page']}] "
                  f"({score_label}sim={ck['score']:.3f})")
            preview = ck["text"][:140].replace("\n", " ")
            print(f"       {preview}{'...' if len(ck['text']) > 140 else ''}")
        print()

    print()
    print(f"  ANSWER:")
    print()
    # Indent each line of the answer by 2 spaces for readability
    for line in answer["answer"].splitlines():
        print(f"  {line}")
    print()

    if answer["citations"] and not answer["is_idk"]:
        print(f"  Distinct sources cited ({len(answer['citations'])}):")
        for c in answer["citations"]:
            print(f"    - {c['pdf_name']} · p.{c['page']}")
        print()

    print("─" * 70)
    rerank_str = f" (rerank +{retrieval['rerank_ms']}ms)" if retrieval["rerank_ms"] else ""
    print(f"  Retrieval:  {retrieval['retrieval_ms']}ms{rerank_str}")
    print(f"  Generation: {answer['generation_ms']}ms"
          f" · {answer['input_tokens']} in, {answer['output_tokens']} out")
    print(f"  TOTAL:      {total_ms}ms")
    if answer["is_idk"]:
        print(f"  Verdict:    IDK (model declined to answer — chunks insufficient)")
    print("─" * 70)


if __name__ == "__main__":
    main()
