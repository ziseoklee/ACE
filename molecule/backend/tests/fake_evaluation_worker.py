"""Test-only fault injection around the real CPU evaluation worker."""

import sys
import time
from pathlib import Path

import ace_backend.evaluation_worker as worker
from evaluation.metrics import druglikeness

root, mode = sys.argv[1:]
if mode == "crash":
    raise SystemExit(3)
if mode == "timeout":
    time.sleep(600)


def fail_metric(*args: object, **kwargs: object) -> float:
    raise RuntimeError("private /private/scoring/error")


if mode == "metric_failure":
    druglikeness.QED.qed = fail_metric
if mode == "write_failure":
    original_write = worker.write_json

    def fail_write(path: Path, value: object) -> None:
        if path.name == "result.json":
            raise OSError("private /private/storage/full")
        original_write(path, value)

    worker.write_json = fail_write

worker.run_evaluation(Path(root))
