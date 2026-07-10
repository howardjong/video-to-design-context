from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

LOGGER_NAME = "tastepack"
_job_id: ContextVar[str] = ContextVar("tastepack_job_id", default="-")
_REDACTION_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+)?)\S+"),
    re.compile(r"(?i)((?:x-goog-api-key|api[_-]?key|gemini_api_key|google_api_key)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"(?i)([?&](?:key|api_key|x-goog-api-key)=)[^&\s]+"),
)


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


class JobContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id.get()
        return True


@contextmanager
def job_log_context(job_id: str):
    token = _job_id.set(job_id)
    try:
        yield
    finally:
        _job_id.reset(token)


def configure_logging(verbosity: str = "normal", log_file: Path | None = None) -> None:
    logger = get_logger()
    logger.handlers.clear()
    logger.propagate = False
    level = logging.DEBUG if verbosity == "debug" else logging.WARNING
    logger.setLevel(logging.DEBUG if log_file else level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.addFilter(JobContextFilter())
    handler.setFormatter(
        RedactingFormatter("%(levelname)s %(name)s [job=%(job_id)s]: %(message)s")
    )
    logger.addHandler(handler)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(JobContextFilter())
        file_handler.setFormatter(
            RedactingFormatter(
                "%(asctime)s %(levelname)s %(name)s [job=%(job_id)s]: %(message)s"
            )
        )
        logger.addHandler(file_handler)


def redact_secrets(message: object) -> str:
    text = str(message)
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text
