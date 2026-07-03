"""Logging configuration with request tracking."""

import logging
import logging.handlers
import uuid
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_FORMAT = "[%(asctime)s] %(levelname)-8s %(request_id)s %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "rag.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.addHandler(_console)
logger.addFilter(RequestIdFilter())


def get_request_id() -> str:
    return uuid.uuid4().hex[:12]
