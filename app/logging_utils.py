"""Shared logging setup: stdout plus a timestamped file under logs/, mp2-style."""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Libraries whose DEBUG output would swamp a debug-level log file.
NOISY_LOGGERS = ("PIL", "pdfminer", "matplotlib", "urllib3")


def setup_logging(
    script_name: str,
    extra_handlers: list[logging.Handler] | None = None,
    console: bool = True,
    debug_file: bool = False,
) -> Path:
    """Log to logs/<YYYYMMDD_HHMMSS>_<script_name>.log, and to stdout unless
    console=False. The console and extra handlers stay at INFO; with
    debug_file=True the file also captures DEBUG records (verbose detail
    such as full extracted text belongs at DEBUG level).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{script_name}.log"

    handlers: list[logging.Handler] = []
    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        handlers.append(stream_handler)
    handlers.append(logging.FileHandler(log_file))
    for handler in extra_handlers or []:
        handler.setLevel(logging.INFO)
        handlers.append(handler)

    level = logging.DEBUG if debug_file else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)
    if debug_file:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)
    return log_file
