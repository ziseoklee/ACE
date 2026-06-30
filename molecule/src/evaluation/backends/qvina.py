import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

PATH_QVINA2 = Path(__file__).parents[2] / "lib" / "qvina2" / "qvina02"
logger = logging.getLogger(__name__)


# ----------------- Data classes (aligned to your Gnina style) -----------------


@dataclass
class QvinaPose:
    mode: int
    affinity: float  # kcal/mol (more negative = better)
    rmsd_lb: float | None = None
    rmsd_ub: float | None = None


@dataclass
class QvinaResult:
    poses: list[QvinaPose]  # sorted by affinity (ascending)
    best: QvinaPose | None  # first item of poses, or None
    raw_stdout: str  # qvina stdout
    raw_stderr: str  # qvina stderr
    cmd: str  # executed command
    pose_mols: list[Mol]  # RDKit molecules for poses (order matches poses)
    best_mol: Mol | None  # RDKit molecule for best pose


# ----------------------------- Helpers ---------------------------------------


def _run_cmd_capture(cmd: list[str], timeout: float = 60.0) -> tuple[str, str, int]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
    return proc.stdout, proc.stderr, proc.returncode


def _require_path_exists(p: Path, what: str):
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"{what} not created: {p}")


def _mol_has_conformer(m: Mol) -> bool:
    try:
        return m is not None and m.GetNumConformers() > 0
    except Exception:
        return False


# -------------------------- I/O conversions -----------------------------------


def ligand_mol_to_pdbqt(mol: Chem.Mol, out_pdbqt: Path) -> None:
    """
    RDKit Mol (with 3D) -> PDBQT using Meeko.
    Compatible with Meeko versions that return either a MoleculeSetup
    or a list of MoleculeSetup from .prepare(mol).
    """
    # save to temporary sdf file
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sdf_file = td / "ligand.sdf"
        writer = Chem.SDWriter(str(sdf_file))
        writer.write(mol)
        writer.close()
        cmd = ["obabel", "-isdf", str(sdf_file), "-opdbqt", "-O", str(out_pdbqt)]
        stdout, stderr, rc = _run_cmd_capture(cmd)
        if rc != 0:
            raise RuntimeError(f"Open Babel failed (ligand):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        _require_path_exists(out_pdbqt, "Ligand PDBQT")
    return


def receptor_pdb_to_pdbqt(pdb_path: Path, out_pdbqt: Path, ph: float = 7.4) -> None:
    """Use Open Babel CLI to create receptor PDBQT (strips waters, adds Hs)."""
    cmd = [
        "obabel",
        "-ipdb",
        str(pdb_path),
        "-opdbqt",
        "-O",
        str(out_pdbqt),
        "-xr",
        "-p",
        str(ph),
    ]
    stdout, stderr, rc = _run_cmd_capture(cmd)
    if rc != 0:
        raise RuntimeError(f"Open Babel failed (receptor):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    _require_path_exists(out_pdbqt, "Receptor PDBQT")


# ------------------------------ Box utils ------------------------------------


def box_from_mol(mol: Mol, pad: float = 8.0) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute a Vina-style box centered on the ligand’s coords with padding."""
    import numpy as np

    if not _mol_has_conformer(mol):
        raise ValueError("Ligand Mol has no 3D conformer for box computation.")
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    cmin, cmax = coords.min(0), coords.max(0)
    center = ((cmin + cmax) / 2.0).tolist()
    size = cmax - cmin + 2.0 * pad
    size = np.maximum(size, 10.0).tolist()  # enforce minimum box
    return tuple(center), tuple(size)


# ------------------------------ Qvina run ------------------------------------

# Vina/Qvina often prints a table:
# mode | affinity | dist from best | rmsd l.b. | rmsd u.b.
_ROW5 = re.compile(r"^\s*(\d+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)")
# Some builds print only 4 cols: mode, affinity, rmsd_l, rmsd_u
_ROW4 = re.compile(r"^\s*(\d+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+([\-0-9.]+)")


def _parse_qvina_stdout(stdout: str) -> list[QvinaPose]:
    poses: list[QvinaPose] = []
    for ln in stdout.splitlines():
        m = _ROW5.match(ln)
        if m:
            poses.append(
                QvinaPose(
                    mode=int(m.group(1)),
                    affinity=float(m.group(2)),
                    rmsd_lb=float(m.group(4)),
                    rmsd_ub=float(m.group(5)),
                )
            )
            continue
        m = _ROW4.match(ln)
        if m:
            poses.append(
                QvinaPose(
                    mode=int(m.group(1)),
                    affinity=float(m.group(2)),
                    rmsd_lb=float(m.group(3)),
                    rmsd_ub=float(m.group(4)),
                )
            )
    poses.sort(key=lambda p: p.affinity)  # most negative first
    return poses


def _pdbqt_to_sdf(pdbqt_in: Path, sdf_out: Path) -> None:
    """Convert Qvina PDBQT poses to SDF with Open Babel to re-load into RDKit."""
    cmd = ["obabel", "-ipdbqt", str(pdbqt_in), "-osdf", "-O", str(sdf_out)]
    stdout, stderr, rc = _run_cmd_capture(cmd)
    if rc != 0:
        raise RuntimeError(f"Open Babel failed (poses PDBQT->SDF):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    _require_path_exists(sdf_out, "Poses SDF")


def _run_qvina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    out_pdbqt: Path,
) -> tuple[str, str, int, list[str]]:
    cx, cy, cz = center
    sx, sy, sz = size
    cmd = [
        str(PATH_QVINA2),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{cx:.3f}",
        "--center_y",
        f"{cy:.3f}",
        "--center_z",
        f"{cz:.3f}",
        "--size_x",
        f"{sx:.3f}",
        "--size_y",
        f"{sy:.3f}",
        "--size_z",
        f"{sz:.3f}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--seed",
        str(seed),
        "--out",
        str(out_pdbqt),
    ]
    stdout, stderr, rc = _run_cmd_capture(cmd)
    return stdout, stderr, rc, cmd


# --------------------------- Public entry point ------------------------------


def qvina_score_from_mol(
    protein_pdb_path: str,
    ligand_mol: Mol,
    *,
    ref_mol: Mol | None = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    pad: float = 8.0,
) -> QvinaResult:
    """
    Run Qvina2 given a protein PDB path and an RDKit Mol (with 3D conformer).
    Returns poses sorted by affinity and RDKit pose molecules (converted via OB).
    """
    if not Path(protein_pdb_path).exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb_path}")
    if not _mol_has_conformer(ligand_mol):
        raise ValueError("Ligand Mol must have a 3D conformer.")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        receptor_pdbqt = td / "receptor.pdbqt"
        ligand_pdbqt = td / "ligand.pdbqt"
        out_pdbqt = td / "qvina_poses.pdbqt"
        out_sdf = td / "qvina_poses.sdf"

        logger.debug("Preparing Qvina inputs.")
        # Prepare inputs
        receptor_pdb_to_pdbqt(Path(protein_pdb_path), receptor_pdbqt)
        ligand_mol_to_pdbqt(ligand_mol, ligand_pdbqt)

        logger.debug("Computing Qvina docking box from ligand coordinates.")
        # Box from ligand coords
        center, size = box_from_mol(ref_mol if ref_mol is not None else ligand_mol, pad=pad)
        # Run qvina
        logger.debug("Running Qvina.")
        stdout, stderr, rc, cmd = _run_qvina(
            receptor_pdbqt,
            ligand_pdbqt,
            center,
            size,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            seed=seed,
            out_pdbqt=out_pdbqt,
        )
        if rc != 0:
            raise RuntimeError(f"Qvina failed with exit code {rc}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

        poses = _parse_qvina_stdout(stdout)
        # Read docked poses back into RDKit
        pose_mols: list[Mol] = []
        if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
            try:
                _pdbqt_to_sdf(out_pdbqt, out_sdf)
                suppl = Chem.SDMolSupplier(str(out_sdf), removeHs=False, sanitize=False)
                pose_mols = [m for m in suppl if m is not None]
            except Exception:
                pose_mols = []

        # Align count if lengths differ
        if pose_mols and len(pose_mols) != len(poses):
            pose_mols = pose_mols[: len(poses)]

        best_mol = pose_mols[0] if pose_mols else None
        return QvinaResult(
            poses=poses,
            best=poses[0] if poses else None,
            raw_stdout=stdout,
            raw_stderr=stderr,
            cmd=" ".join(map(str, cmd)),
            pose_mols=pose_mols,
            best_mol=best_mol,
        )
