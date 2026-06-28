import contextlib
import io
import logging
from collections.abc import Iterator


class LoggerWriter(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.logger.log(self.level, self.buffer.rstrip())
        self.buffer = ""


@contextlib.contextmanager
def redirect_output_to_logger(logger: logging.Logger) -> Iterator[None]:
    stdout = LoggerWriter(logger, logging.INFO)
    stderr = LoggerWriter(logger, logging.WARNING)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()
