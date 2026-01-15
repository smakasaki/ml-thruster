import sys
from pathlib import Path

from loguru import logger

DEFAULT_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL = "DEBUG"
LOG_ROTATION_SIZE = "10 MB"
LOG_LEVEL_PADDING = "<8"

CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    f"<level>{{level: {LOG_LEVEL_PADDING}}}</level> | "
    "<level>{message}</level>"
)
FILE_LOG_FORMAT = f"{{time:YYYY-MM-DD HH:mm:ss}} | {{level: {LOG_LEVEL_PADDING}}} | {{message}}"


def setup_logger(enable_file_logging: bool = False, log_path: Path | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format=CONSOLE_LOG_FORMAT,
        level=DEFAULT_LOG_LEVEL,
    )

    if enable_file_logging:
        if log_path is None:
            msg = "log_path must be provided when enable_file_logging is True"
            raise ValueError(msg)
        logger.add(
            log_path,
            format=FILE_LOG_FORMAT,
            level=FILE_LOG_LEVEL,
            rotation=LOG_ROTATION_SIZE,
        )
        logger.info(f"Logging to file: {log_path}")
