"""
audit.py — Merkle hash-chained audit log.

Pattern ported from Atticus / Bearpaw (`app/core/audit.py`).
Reimplemented from scratch to keep this repo self-contained.

Every action that touches user data — incoming chat message, guardrail
decision, retrieval, generation, ingestion start/complete — is written
as a JSON line in data/audit.jsonl with:

    timestamp, action_type, details, prev_hash, hash

Each entry's `hash` is SHA-256 over its own fields including `prev_hash`,
so the chain links cryptographically. Tampering with any past entry
breaks every subsequent hash. `verify()` walks the file and confirms
the integrity end-to-end.

Genesis hash: sha256(b"NAVGURUKUL_GENESIS_BLOCK") — fixed constant.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MerkleAuditChain:
    """Append-only hash-chained log on disk."""

    GENESIS_HASH = hashlib.sha256(b"NAVGURUKUL_GENESIS_BLOCK").hexdigest()

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Walk to the end of the file and return the last entry's hash."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return self.GENESIS_HASH
        # Backward-walk to the last newline to find the final line cheaply
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                # Skip trailing whitespace
                pos = end
                while pos > 0:
                    pos -= 1
                    f.seek(pos)
                    ch = f.read(1)
                    if ch == b"\n" and pos != end - 1:
                        break
                last_line = f.readline().decode("utf-8", errors="replace").strip()
                if not last_line:
                    # File ends in blank lines — rewind further
                    f.seek(0)
                    last_line = f.readlines()[-1].decode("utf-8", errors="replace").strip()
                if last_line:
                    return json.loads(last_line).get("hash", self.GENESIS_HASH)
        except Exception as exc:
            logger.warning(f"[audit] could not read last hash: {exc!r}; resetting to genesis")
        return self.GENESIS_HASH

    @staticmethod
    def _hash_data(data: Dict[str, Any]) -> str:
        s = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(s).hexdigest()

    def log(self, action_type: str, details: Dict[str, Any]) -> str:
        """Append one entry to the chain. Returns this entry's hash."""
        entry_data = {
            "timestamp":   time.time(),
            "action_type": action_type,
            "details":     details,
            "prev_hash":   self._last_hash,
        }
        h = self._hash_data(entry_data)
        line = {**entry_data, "hash": h}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._last_hash = h
        return h

    def verify(self) -> Dict[str, Any]:
        """Walk the file and confirm every hash links + matches its data."""
        if not self.log_path.exists():
            return {"valid": True, "entries": 0, "broken_at": None}
        prev = self.GENESIS_HASH
        count = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    return {"valid": False, "entries": count, "broken_at": i,
                            "reason": "non-JSON line"}
                if entry.get("prev_hash") != prev:
                    return {"valid": False, "entries": count, "broken_at": i,
                            "reason": "prev_hash mismatch"}
                data = {k: entry[k] for k in ("timestamp", "action_type", "details", "prev_hash")
                        if k in entry}
                if self._hash_data(data) != entry.get("hash"):
                    return {"valid": False, "entries": count, "broken_at": i,
                            "reason": "hash mismatch"}
                prev = entry["hash"]
                count += 1
        return {"valid": True, "entries": count, "broken_at": None}

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent N entries (newest first), with short hashes for display."""
        if not self.log_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        tail = entries[-limit:]
        tail.reverse()  # newest first
        return tail


# ── Module-level singleton ───────────────────────────────────────────────────

_chain: Optional[MerkleAuditChain] = None


def init_chain(log_path: Path) -> MerkleAuditChain:
    global _chain
    if _chain is None:
        _chain = MerkleAuditChain(log_path)
    return _chain


def get_chain() -> Optional[MerkleAuditChain]:
    return _chain


def log_action(action_type: str, details: Dict[str, Any]) -> Optional[str]:
    """Convenience: no-op if the chain hasn't been initialized yet."""
    if _chain is None:
        return None
    try:
        return _chain.log(action_type, details)
    except Exception as exc:
        logger.error(f"[audit] log failed for {action_type}: {exc!r}")
        return None
