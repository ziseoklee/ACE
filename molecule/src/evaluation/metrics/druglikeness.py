import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski, RDConfig
from rdkit.Chem.rdchem import Mol

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except ImportError:
    logger.warning("RDKit SA_Score sascorer is not available. SA score will be 0.0.")
    sascorer = None


def evaluate_druglikeness(mol: Mol | None) -> dict[str, float]:
    scores = {"validity": 0.0, "QED": 0.0, "SA": 0.0, "LogP": 0.0, "Lipinski": 0.0}
    if mol is None:
        return scores

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        scores["validity"] = 1.0
    except Exception:
        return scores

    scores["QED"] = _safe_float(lambda: QED.qed(mol_copy))
    scores["SA"] = _safe_sa_score(mol_copy)
    scores["LogP"] = _safe_float(lambda: Descriptors.MolLogP(mol_copy))
    scores["Lipinski"] = _safe_lipinski_score(mol_copy, scores["LogP"])
    return scores


def evaluate_ligand_sdf(ligand_sdf: Path) -> dict[str, float]:
    mol = load_ligand_sdf(ligand_sdf)
    return evaluate_druglikeness(mol)


def load_ligand_sdf(ligand_sdf: Path) -> Mol | None:
    if not ligand_sdf.exists():
        return None
    supplier = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False)
    return supplier[0] if len(supplier) > 0 else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligand_sdf", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args(argv)

    result: dict[str, Any] = {"ligand_sdf": str(args.ligand_sdf)}
    result.update(evaluate_ligand_sdf(args.ligand_sdf))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")
    print(payload)


def _safe_float(fn) -> float:
    try:
        return float(fn())
    except Exception:
        return 0.0


def _safe_sa_score(mol: Mol) -> float:
    if sascorer is None:
        return 0.0
    try:
        raw_sa = float(sascorer.calculateScore(mol))
        # Normalize to [0, 1] as per DiffSBDD
        # Raw scale: 1 (easy) to 10 (hard)
        # Normalized: 1.0 (easy) to 0.0 (hard)
        # Clamp to [0, 1] just in case raw score drifts slightly outside 1-10
        return max(0.0, min(1.0, (10.0 - raw_sa) / 9.0))
    except Exception:
        return 0.0


def _safe_lipinski_score(mol: Mol, logp: float) -> float:
    # Lipinski Rule of 5
    # MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
    try:
        violations = 0
        violations += int(Descriptors.MolWt(mol) > 500)
        violations += int(logp > 5)
        violations += int(Lipinski.NumHDonors(mol) > 5)
        violations += int(Lipinski.NumHAcceptors(mol) > 10)
        return (5 - violations) / 5.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
