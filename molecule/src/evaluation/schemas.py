from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrossDockedSampleRecord:
    task_id: str
    sample_id: int
    ligand_sdf_path: Path
    task_dir: Path
    exists: bool
