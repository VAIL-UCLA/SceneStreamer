from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings


class _SuppressKnownStartupMessages(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Summary file is not found at" not in message


_STARTUP_CONFIGURED = False


def configure_startup_noise_filters() -> None:
    global _STARTUP_CONFIGURED
    if _STARTUP_CONFIGURED:
        return

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    suppress_filter = _SuppressKnownStartupMessages()
    for logger_name in ("MetaDrive", "metadrive", ""):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(existing, _SuppressKnownStartupMessages) for existing in logger.filters):
            logger.addFilter(suppress_filter)

    _STARTUP_CONFIGURED = True


@contextlib.contextmanager
def quiet_native_startup_noise():
    configure_startup_noise_filters()

    if sys.platform != "darwin":
        yield
        return

    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
