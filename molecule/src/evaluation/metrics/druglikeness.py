import logging
import math
import os
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import QED, RDConfig, rdMolDescriptors
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
    scores["LogP"] = _safe_float(lambda: rdMolDescriptors.CalcCrippenDescriptors(mol_copy)[0])
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


def _safe_float(fn) -> float:
    try:
        return float(fn())
    except Exception:
        return 0.0


def calculate_sa_normalized(mol: Mol) -> float:
    """Calculate the existing normalized SA score, preserving calculation failures."""
    if sascorer is None:
        raise ImportError("RDKit SA_Score is unavailable.")
    raw_sa = float(sascorer.calculateScore(mol))
    if not math.isfinite(raw_sa):
        raise ValueError("SA score must be finite.")
    # Normalize the raw 1 (easy) to 10 (hard) scale as in DiffSBDD.
    return max(0.0, min(1.0, (10.0 - raw_sa) / 9.0))


def calculate_lipinski_legacy(mol: Mol, logp: float) -> float:
    """The CLI's four-rule score with its historical denominator of five."""
    weight = float(rdMolDescriptors._CalcMolWt(mol))
    donors = float(rdMolDescriptors.CalcNumHBD(mol))
    acceptors = float(rdMolDescriptors.CalcNumHBA(mol))
    if not all(math.isfinite(value) for value in (weight, logp, donors, acceptors)):
        raise ValueError("Lipinski inputs must be finite.")
    violations = 0
    violations += int(weight > 500)
    violations += int(logp > 5)
    violations += int(donors > 5)
    violations += int(acceptors > 10)
    return (5 - violations) / 5.0


def _safe_sa_score(mol: Mol) -> float:
    try:
        return calculate_sa_normalized(mol)
    except Exception:
        return 0.0


def _safe_lipinski_score(mol: Mol, logp: float) -> float:
    try:
        return calculate_lipinski_legacy(mol, logp)
    except Exception:
        return 0.0
