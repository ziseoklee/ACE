import ast
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from evaluation.io import iter_crossdocked_sample_records
from evaluation.metrics.docking import evaluate_docking
from evaluation.metrics.druglikeness import evaluate_druglikeness
from evaluation.summarize import summarize_samples

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = {"druglikeness", "docking"}


@dataclass(frozen=True)
class CrossDockedTaskMetadata:
    task_id: str
    protein_filename: str
    protein_pdb_path: Path
    ref_length: int
    ref_ligand: Mol
    reference_affinity: float | None


@dataclass(frozen=True)
class CrossDockedEvaluationResult:
    sample_df: pd.DataFrame
    task_df: pd.DataFrame
    summary_df: pd.DataFrame


def evaluate_crossdocked_run(
    run_dir: Path,
    data_root: Path,
    *,
    metrics: Sequence[str] = ("druglikeness",),
    expected_num_samples: int | None = None,
    output_dir: Path | None = None,
    max_ligand_atoms: int = 29,
    docking_exhaustiveness: int = 8,
    docking_num_modes: int = 9,
    docking_seed: int = 42,
    docking_pad: float = 8.0,
) -> CrossDockedEvaluationResult:
    metrics = _normalize_metrics(metrics)
    output_dir = output_dir or run_dir / "evaluation"
    reference_affinities = load_reference_affinities(data_root)
    sample_records = list(iter_crossdocked_sample_records(run_dir, expected_num_samples=expected_num_samples))

    rows: list[dict[str, Any]] = []
    metadata_cache: dict[str, CrossDockedTaskMetadata] = {}
    for record in tqdm(sample_records, desc="Evaluating CrossDocked samples", unit="sample"):
        try:
            metadata = metadata_cache.setdefault(
                record.task_id,
                load_crossdocked_task_metadata(data_root, record.task_id, reference_affinities),
            )
        except Exception as exc:
            logger.exception("Failed to load CrossDocked2020 metadata for task %s", record.task_id)
            rows.append(_metadata_failure_row(record, exc))
            continue

        if metadata.ref_length > max_ligand_atoms:
            continue

        ligand_mol, ligand_error = _load_single_fragment_ligand(record.ligand_sdf_path)
        row: dict[str, Any] = {
            "task_id": record.task_id,
            "sample_id": record.sample_id,
            "ligand_sdf_path": str(record.ligand_sdf_path),
            "exists": record.exists,
            "single_fragment": ligand_mol is not None,
            "ligand_error": ligand_error,
            "protein_filename": metadata.protein_filename,
            "protein_pdb_path": str(metadata.protein_pdb_path),
            "ref_length": metadata.ref_length,
            "reference_affinity": metadata.reference_affinity,
        }

        if "druglikeness" in metrics:
            row.update(evaluate_druglikeness(ligand_mol))

        if "docking" in metrics:
            row.update(
                evaluate_docking(
                    metadata.protein_pdb_path,
                    ligand_mol,
                    exhaustiveness=docking_exhaustiveness,
                    num_modes=docking_num_modes,
                    seed=docking_seed,
                    pad=docking_pad,
                )
            )

        rows.append(row)

    sample_df = pd.DataFrame(rows)
    task_df, summary_df = summarize_samples(sample_df)

    write_evaluation_frames(
        CrossDockedEvaluationResult(sample_df=sample_df, task_df=task_df, summary_df=summary_df),
        output_dir,
    )

    return CrossDockedEvaluationResult(sample_df=sample_df, task_df=task_df, summary_df=summary_df)


def write_evaluation_frames(result: CrossDockedEvaluationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.sample_df.to_csv(output_dir / "samples.csv", index=False)
    result.task_df.to_csv(output_dir / "tasks.csv", index=False)
    result.summary_df.to_csv(output_dir / "summary.csv", index=False)


def load_crossdocked_task_metadata(
    data_root: Path,
    task_id: str,
    reference_affinities: dict[str, float] | None = None,
) -> CrossDockedTaskMetadata:
    processed_path = data_root / "processed" / f"{task_id}.pt"
    data = torch.load(processed_path, weights_only=False)
    protein_filename = data["protein_filename"]
    ref_ligand = data["mol"]
    return CrossDockedTaskMetadata(
        task_id=task_id,
        protein_filename=protein_filename,
        protein_pdb_path=data_root / "crossdocked_pocket10" / protein_filename,
        ref_length=ref_ligand.GetNumAtoms(),
        ref_ligand=ref_ligand,
        reference_affinity=(reference_affinities or {}).get(task_id),
    )


def load_reference_affinities(data_root: Path) -> dict[str, float]:
    len_dict_path = data_root / "len_dict.csv"
    if not len_dict_path.exists():
        return {}

    df = pd.read_csv(len_dict_path)
    affinities = {}
    for _, row in df.iterrows():
        try:
            payload = ast.literal_eval(row["length"])
            affinities[str(row["index"])] = float(payload["affinity"])
        except Exception:
            continue
    return affinities


def _normalize_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(metric.lower() for metric in metrics)
    unsupported = sorted(set(normalized) - SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"Unsupported evaluation metrics: {unsupported}. Supported metrics: {sorted(SUPPORTED_METRICS)}")
    return normalized


def _load_single_fragment_ligand(ligand_sdf_path: Path) -> tuple[Mol | None, str]:
    if not ligand_sdf_path.exists():
        return None, "missing_ligand_sdf"

    try:
        supplier = Chem.SDMolSupplier(str(ligand_sdf_path), sanitize=False)
        mol = supplier[0] if len(supplier) > 0 else None
    except Exception as exc:
        return None, f"read_failed: {exc}"

    if mol is None:
        return None, "read_failed"

    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            return None, "not_single_fragment"
    except Exception as exc:
        return None, f"fragment_check_failed: {exc}"

    return mol, ""


def _metadata_failure_row(record, exc: Exception) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "sample_id": record.sample_id,
        "ligand_sdf_path": str(record.ligand_sdf_path),
        "exists": record.exists,
        "single_fragment": False,
        "ligand_error": "",
        "metadata_error": str(exc),
    }
