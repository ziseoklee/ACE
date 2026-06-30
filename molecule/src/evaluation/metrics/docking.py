from pathlib import Path
from typing import Any

from rdkit.Chem.rdchem import Mol

from evaluation.backends.qvina import qvina_score_from_mol
from evaluation.metrics.druglikeness import load_ligand_sdf


def evaluate_docking(
    pocket_pdb: Path,
    ligand_mol: Mol | None,
    *,
    ref_mol: Mol | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    pad: float = 8.0,
) -> dict[str, Any]:
    if ligand_mol is None:
        return _failed_docking_result("Ligand molecule could not be loaded.")

    try:
        result = qvina_score_from_mol(
            str(pocket_pdb),
            ligand_mol,
            ref_mol=ref_mol,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            seed=seed,
            pad=pad,
        )
    except Exception as exc:
        return _failed_docking_result(str(exc))

    if result.best is None:
        return _failed_docking_result("Qvina returned no valid pose.")

    return {
        "docking_success": 1.0,
        "qvina_affinity": float(result.best.affinity),
        "qvina_num_poses": len(result.poses),
        "qvina_cmd": result.cmd,
        "docking_error": "",
    }


def evaluate_docking_sdf(
    pocket_pdb: Path,
    ligand_sdf: Path,
    *,
    ref_ligand_sdf: Path | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    pad: float = 8.0,
) -> dict[str, Any]:
    ligand_mol = load_ligand_sdf(ligand_sdf)
    ref_mol = load_ligand_sdf(ref_ligand_sdf) if ref_ligand_sdf is not None else None
    return evaluate_docking(
        pocket_pdb,
        ligand_mol,
        ref_mol=ref_mol,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        seed=seed,
        pad=pad,
    )


def _failed_docking_result(error: str) -> dict[str, Any]:
    return {
        "docking_success": 0.0,
        "qvina_affinity": 0.0,
        "qvina_num_poses": 0,
        "qvina_cmd": "",
        "docking_error": error,
    }
