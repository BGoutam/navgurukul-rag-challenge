#!/usr/bin/env python3
"""
serve.py — Launch the FastAPI server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080
    python scripts/serve.py --reload     # auto-reload on file changes
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_HERE))


def main():
    parser = argparse.ArgumentParser(description="Launch the Navgurukul RAG server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on file changes (dev mode).")
    args = parser.parse_args()

    import uvicorn
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Open your browser to that URL to use the chat UI.")
    print()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
