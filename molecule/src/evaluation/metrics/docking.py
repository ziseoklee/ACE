import argparse
import json
from pathlib import Path
from typing import Any

from rdkit.Chem.rdchem import Mol

from evaluation.metrics.druglikeness import load_ligand_sdf
from utils.metrics.qvina import qvina_score_from_mol


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket_pdb", type=Path, required=True)
    parser.add_argument("--ligand_sdf", type=Path, required=True)
    parser.add_argument("--ref_ligand_sdf", type=Path, default=None)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad", type=float, default=8.0)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args(argv)

    result = {
        "pocket_pdb": str(args.pocket_pdb),
        "ligand_sdf": str(args.ligand_sdf),
        "ref_ligand_sdf": str(args.ref_ligand_sdf) if args.ref_ligand_sdf is not None else None,
    }
    result.update(
        evaluate_docking_sdf(
            args.pocket_pdb,
            args.ligand_sdf,
            ref_ligand_sdf=args.ref_ligand_sdf,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            seed=args.seed,
            pad=args.pad,
        )
    )

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")
    print(payload)


def _failed_docking_result(error: str) -> dict[str, Any]:
    return {
        "docking_success": 0.0,
        "qvina_affinity": 0.0,
        "qvina_num_poses": 0,
        "qvina_cmd": "",
        "docking_error": error,
    }


if __name__ == "__main__":
    main()
