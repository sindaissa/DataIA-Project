"""ADOS Structured JSON Logging with correlation IDs."""
from __future__ import annotations
import json, logging, sys, uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from ados.config import get_settings, LOGS_DIR

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="no-trace")

def set_correlation_id(cid: Optional[str] = None) -> str:
    cid = cid or str(uuid.uuid4())[:12]
    _correlation_id.set(cid)
    return cid

def get_correlation_id() -> str:
    return _correlation_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "cid": get_correlation_id(),
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry["extra"] = record.extra_data
        return json.dumps(entry, ensure_ascii=False, default=str)


_configured = False

def get_logger(name: str) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
        root = logging.getLogger()
        root.setLevel(level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(JsonFormatter())
        root.addHandler(ch)
        try:
            fh = logging.FileHandler(LOGS_DIR / "ados.log", encoding="utf-8")
            fh.setFormatter(JsonFormatter())
            root.addHandler(fh)
        except Exception:
            pass
        _configured = True
    return logger
