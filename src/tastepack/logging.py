from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOGGER_NAME = "tastepack"


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def configure_logging(verbosity: str = "normal", log_file: Path | None = None) -> None:
    logger = get_logger()
    logger.handlers.clear()
    logger.propagate = False
    level = logging.DEBUG if verbosity == "debug" else logging.WARNING
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)


def redact_secrets(message: object) -> str:
    text = str(message)
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text
