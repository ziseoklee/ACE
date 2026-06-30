import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from evaluation.schemas import CrossDockedSampleRecord

_SAMPLE_SDF_RE = re.compile(r"^(?P<sample_id>\d+)\.sdf$")


def iter_crossdocked_sample_records(
    run_dir: Path,
    expected_num_samples: int | None = None,
) -> Iterable[CrossDockedSampleRecord]:
    """Yield generated ligand SDF records from a CrossDocked2020 run directory."""
    inference_dir = resolve_crossdocked_inference_dir(run_dir)
    for task_dir in _iter_task_dirs(inference_dir):
        yield from _iter_task_sample_records(task_dir, expected_num_samples=expected_num_samples)


def read_crossdocked_sample_index(
    run_dir: Path,
    expected_num_samples: int | None = None,
) -> pd.DataFrame:
    records = list(iter_crossdocked_sample_records(run_dir, expected_num_samples=expected_num_samples))
    rows = [
        {
            "task_id": record.task_id,
            "sample_id": record.sample_id,
            "ligand_sdf_path": str(record.ligand_sdf_path),
            "task_dir": str(record.task_dir),
            "exists": record.exists,
        }
        for record in records
    ]
    return pd.DataFrame(rows)


def resolve_crossdocked_inference_dir(run_dir: Path) -> Path:
    if not run_dir.exists():
        raise FileNotFoundError(f"CrossDocked2020 run directory does not exist: {run_dir}")

    inference_dir = run_dir / "inference"
    if not inference_dir.exists():
        raise FileNotFoundError(
            "CrossDocked2020 evaluation expects generated samples under "
            f"`run_dir/inference`, but the directory does not exist: {inference_dir}"
        )
    return inference_dir


def _iter_task_dirs(inference_dir: Path) -> list[Path]:
    if not inference_dir.exists():
        raise FileNotFoundError(f"CrossDocked2020 inference directory does not exist: {inference_dir}")
    task_dirs = [path for path in inference_dir.iterdir() if path.is_dir() and _is_task_dir(path)]
    return sorted(task_dirs, key=lambda path: _task_sort_key(path.name))


def _iter_task_sample_records(
    task_dir: Path,
    expected_num_samples: int | None,
) -> Iterable[CrossDockedSampleRecord]:
    if expected_num_samples is not None:
        sample_ids = list(range(expected_num_samples))
    else:
        sample_ids = _existing_sample_ids(task_dir)

    for sample_id in sample_ids:
        ligand_sdf_path = task_dir / f"{sample_id}.sdf"
        yield CrossDockedSampleRecord(
            task_id=task_dir.name,
            sample_id=sample_id,
            ligand_sdf_path=ligand_sdf_path,
            task_dir=task_dir,
            exists=ligand_sdf_path.exists(),
        )


def _existing_sample_ids(task_dir: Path) -> list[int]:
    sample_ids = []
    for path in task_dir.iterdir():
        if not path.is_file():
            continue
        match = _SAMPLE_SDF_RE.match(path.name)
        if match is not None:
            sample_ids.append(int(match.group("sample_id")))
    return sorted(sample_ids)


def _is_task_dir(path: Path) -> bool:
    return path.name.isdigit()


def _task_sort_key(task_id: str) -> tuple[int, str]:
    if task_id.isdigit():
        return int(task_id), task_id
    return 10**12, task_id
