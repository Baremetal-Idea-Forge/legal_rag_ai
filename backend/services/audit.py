"""
Audit log — a compliance artifact, not telemetry.

One JSON object per line, append-only, written BEFORE the response leaves the
service. The record carries everything needed to reconstruct how an answer was
produced: the query, the retrieval plan, the chunks (with their character
spans), the answer, its verification scores, and the model that produced it.

A failed write is logged loudly but does not fail the user's request — in this
deployment losing one audit line is better than converting a disk hiccup into
an outage. Revisit that tradeoff before any regulated production use.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only JSONL writer."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record(self, entry: dict[str, Any]) -> None:
        """Append one record, stamped with UTC time."""
        record = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Audit write failed — record lost: %s", record.get("query"))
