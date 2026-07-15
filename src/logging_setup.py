"""
Structured logging configuration shared by every agent and script.

Each log line carries a timestamp, level, logger name (agent name) and
message, and is written both to stdout and to a rotating file so that a
full run can be audited after the fact.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: str = "artifacts/logs", log_file: str = "pipeline.log") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
