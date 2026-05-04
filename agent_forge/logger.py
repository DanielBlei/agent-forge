import logging
import sys

_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "FATAL",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


class _DevFormatter(logging.Formatter):
    def __init__(self, colorize: bool = False) -> None:
        super().__init__()
        self._colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        level = _LEVEL_NAMES.get(record.levelno, "INFO")
        name = record.name
        msg = record.getMessage()

        if self._colorize:
            color = _COLORS.get(record.levelno, "")
            line = f"{_DIM}{ts}{_RESET}  {color}{_BOLD}{level:<5}{_RESET}  {_BOLD}{name}{_RESET}  {msg}"
        else:
            line = f"{ts}  {level:<5}  {name}  {msg}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line



def set_log_level(level: int) -> None:
    """Set level on all agent_forge loggers — call once after arg parsing."""
    for obj_name, obj in logging.Logger.manager.loggerDict.items():
        if obj_name.startswith("agent_forge") and isinstance(obj, logging.Logger):
            obj.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_DevFormatter(colorize=sys.stdout.isatty()))
    logger.addHandler(handler)
    logger.propagate = False

    return logger
