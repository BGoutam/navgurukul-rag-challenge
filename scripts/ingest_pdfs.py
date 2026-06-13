#!/usr/bin/env python3
"""
ingest_pdfs.py — CLI: ingest a directory of PDFs into the RAG ChromaDB.

Usage:
    python scripts/ingest_pdfs.py <pdf_dir>
    python scripts/ingest_pdfs.py <pdf_dir> --force         # rebuild
    python scripts/ingest_pdfs.py <pdf_dir> --db <path>     # custom DB path

Examples:
    # Default: ingest data/pdfs/ into data/chroma_db/
    python scripts/ingest_pdfs.py data/pdfs/

    # Ingest from the Bearpaw corpus directly (no need to copy files)
    python scripts/ingest_pdfs.py ../Bearpaw/Transformers/
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Project root on PYTHONPATH so `app.ingest` imports cleanly
_HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_HERE))

from app.ingest import ingest_directory  # noqa: E402


def _human_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG ChromaDB.")
    parser.add_argument("pdf_dir", type=Path, help="Directory containing PDFs (recursively scanned).")
    parser.add_argument("--db", type=Path, default=_HERE / "data" / "chroma_db",
                        help="ChromaDB persist path (default: data/chroma_db/).")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if a PDF's SHA-256 matches the manifest.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-page progress output.")
    args = parser.parse_args()

    if not args.pdf_dir.exists() or not args.pdf_dir.is_dir():
        sys.exit(f"Error: '{args.pdf_dir}' is not a directory.")

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")

    # ── Light progress reporter ─────────────────────────────────────────────
    last_file = {"name": None, "start": 0.0}

    def progress(pdf_name, page=None, total=None, status=None, reason=None):
        if status == "SKIP":
            print(f"  [skip] {pdf_name}  ({reason})")
            return
        if pdf_name != last_file["name"]:
            if last_file["name"] is not None:
                elapsed = time.time() - last_file["start"]
                print(f"      ({elapsed:.1f}s)")
            last_file["name"] = pdf_name
            last_file["start"] = time.time()
            print(f"  [ingest] {pdf_name}", end="")
        if not args.quiet and page is not None and total:
            # Print a dot every 25 pages, or on the last page
            if page % 25 == 0 or page == total:
                print(f" .{page}/{total}", end="", flush=True)

    print(f"Ingesting from: {args.pdf_dir}")
    print(f"ChromaDB at:    {args.db}")
    if args.force:
        print("Mode:           FORCE re-ingest (ignoring manifest)")
    print()
    t0 = time.time()

    stats = ingest_directory(
        pdf_dir=args.pdf_dir,
        chroma_db_path=args.db,
        force=args.force,
        progress=progress,
    )

    if last_file["name"] is not None:
        elapsed = time.time() - last_file["start"]
        print(f"      ({elapsed:.1f}s)")

    wall = time.time() - t0
    print()
    print("─" * 60)
    print(f"  PDFs total:         {stats['pdfs_total']}")
    print(f"  Ingested:           {stats['pdfs_ingested']}")
    print(f"  Skipped (no change):{stats['pdfs_skipped']}")
    if stats["pdfs_error"]:
        print(f"  Errors:             {stats['pdfs_error']}")
    print(f"  Pages processed:    {stats['total_pages']:,}")
    print(f"  Chunks created:     {stats['total_chunks']:,}")
    print(f"  Collection size:    {stats['collection_count']:,} vectors")
    print(f"  Wall time:          {wall:.1f}s")

    # Approximate disk footprint of the ChromaDB
    if args.db.exists():
        total_bytes = sum(p.stat().st_size for p in args.db.rglob("*") if p.is_file())
        print(f"  ChromaDB on disk:   {_human_size(total_bytes)}")
    print("─" * 60)


if __name__ == "__main__":
    main()
