"""services/audit.py — append-only JSONL audit log."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.audit import AuditLog


def test_record_appends_jsonl_with_timestamp(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record({"query": "q1", "abstained": False})
    log.record({"query": "q2", "abstained": True})

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["query"] == "q1"
    assert second["query"] == "q2"
    assert "ts" in first and first["ts"].endswith("+00:00")  # UTC-stamped


def test_record_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "audit.jsonl"
    AuditLog(path).record({"query": "q"})
    assert path.exists()


def test_record_swallows_write_failure(tmp_path):
    # Parent "directory" is actually a file → OSError inside record().
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    AuditLog(blocker / "audit.jsonl").record({"query": "q"})  # must not raise
