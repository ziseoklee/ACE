import os
import re
import json
import tempfile
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from rdkit import Chem
from rdkit.Chem.rdchem import Mol


@dataclass
class GninaPose:
    mode: int
    affinity: float          # kcal/mol (negative is better)
    intramol: Optional[float]  # kcal/mol (may be None if not present)
    cnn_pose: Optional[float]
    cnn_affinity: Optional[float]


@dataclass
class GninaResult:
    poses: List[GninaPose]           # sorted by affinity (ascending)
    best: Optional[GninaPose]        # first item of poses, or None
    raw_stdout: str                  # for debugging
    raw_stderr: str                  # for debugging
    cmd: str                         # executed command line
    pose_mols: List[Mol]             # RDKit molecules for poses (order matches poses)
    best_mol: Optional[Mol]          # RDKit molecule for best pose


def _run_cmd_capture(cmd: List[str]) -> Tuple[str, str, int]:
    """Run a command and capture stdout, stderr, and returncode."""
    proc = subprocess.run(
        cmd, text=True, capture_output=True, check=False
    )
    return proc.stdout, proc.stderr, proc.returncode


_TABLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s*$"
)

def _parse_gnina_table(stdout: str) -> List[GninaPose]:
    """
    Parse gnina's result table from stdout.
    Returns a list of GninaPose (unsorted).
    Handles the typical header:

        mode |  affinity  |  intramol  |    CNN     |   CNN
             | (kcal/mol) | (kcal/mol) | pose score | affinity
        -----+------------+------------+------------+----------
            1       -6.83 ...
    """
    # breakpoint()
    lines = stdout.splitlines()

    # Find the header separator line (-----+----...)
    start_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^-+\+?-+", ln.strip().replace(" ", "")) or ln.strip().startswith("-----+"):
            start_idx = i
            break
    if start_idx is None:
        # Some modes (e.g., --score_only) print a single line; handle later
        # Try to parse any row-looking lines anyway.
        start_idx = 0

    poses: List[GninaPose] = []
    for ln in lines[start_idx:]:
        m = _TABLE_ROW_RE.match(ln)
        if not m:
            continue
        mode = int(m.group(1))
        affinity = float(m.group(2))
        intramol = float(m.group(3))
        cnn_pose = float(m.group(4))
        cnn_aff = float(m.group(5))
        poses.append(GninaPose(mode, affinity, intramol, cnn_pose, cnn_aff))

    # Fallback: try to parse --score_only output lines like:
    # "Affinity: -7.3 (kcal/mol)"
    if not poses:
        m = re.search(r"Affinity:\s*([-\d\.]+)", stdout)
        if m:
            aff = float(m.group(1))
            poses.append(GninaPose(mode=1, affinity=aff, intramol=None, cnn_pose=None, cnn_affinity=None))

    return poses


def gnina_score(
    pdb_path: str,
    rdkit_mol,
    gnina_path: str = "gnina",
    fix_pose: bool = False,
    seed: int = 42,
    cpu: int = 1,
    extra_args: Optional[List[str]] = None,
) -> GninaResult:
    """
    Run gnina, capture output, parse the score table, and return sorted results.

    - If fix_pose=False: uses --autobox_ligand on the same ligand.
    - If fix_pose=True: uses --score_only (no sampling).
    - Returns poses sorted by affinity (ascending).
    - Also returns RDKit molecules for the docked poses parsed from SDF output.
    """
    extra_args = extra_args or []

    # Write ligand to a safe temp file
    with tempfile.TemporaryDirectory() as td:
        lig_path = Path(td) / "tmp_ligand.pdb"
        # Request SDF output for robust multi-pose parsing
        out_path = Path(td) / "tmp_out.sdf"
        Chem.MolToPDBFile(rdkit_mol, str(lig_path))
        # breakpoint()
        if not fix_pose:
            cmd = [
                gnina_path,
                "-r", str(pdb_path),
                "-l", str(lig_path),
                "-o", str(out_path),
                "--autobox_ligand", str(lig_path),
                "--seed", str(seed),
                "--cpu", str(cpu),
            ] + extra_args
        else:
            cmd = [
                gnina_path,
                "--score_only",
                "-r", str(pdb_path),
                "-l", str(lig_path),
                "-o", str(out_path),
                "--seed", str(seed),
                "--cpu", str(cpu),
            ] + extra_args

        stdout, stderr, rc = _run_cmd_capture(cmd)
        # breakpoint()
        poses = _parse_gnina_table(stdout)

        # Sort by affinity ascending (more negative is better)
        poses.sort(key=lambda p: p.affinity)

        # Parse SDF output into RDKit molecules (one per pose when available)
        pose_mols: List[Mol] = []
        if out_path.exists() and out_path.stat().st_size > 0:
            try:
                suppl = Chem.SDMolSupplier(str(out_path), removeHs=False, sanitize=False)
                pose_mols = [m for m in suppl if m is not None]
            except Exception:
                pose_mols = []

        # Align molecule count with parsed poses if lengths differ
        if pose_mols and len(pose_mols) != len(poses):
            pose_mols = pose_mols[: len(poses)]

        best_mol: Optional[Mol] = pose_mols[0] if pose_mols else None
        # Fallback for score_only: use the input molecule as best if no output mols
        if fix_pose and best_mol is None:
            try:
                best_mol = Chem.Mol(rdkit_mol)
            except Exception:
                best_mol = None

        return GninaResult(
            poses=poses,
            best=poses[0] if poses else None,
            raw_stdout=stdout,
            raw_stderr=stderr,
            cmd=" ".join(cmd),
            pose_mols=pose_mols,
            best_mol=best_mol,
        )

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True)
    parser.add_argument("--ligand_dir", type=str, required=True)
    parser.add_argument("--fix_pose", action="store_true")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--postfix", type=str, default="")
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    pdb_path = args.pdb_path
    ligand_dir = args.ligand_dir
    fix_pose = args.fix_pose
    print(f'fixe pose: {fix_pose}')
    num_samples = args.num_samples
    prefix = args.prefix
    postfix = args.postfix
    for i in range(num_samples):
        try:
            sample = list(Chem.SDMolSupplier(f'{ligand_dir}/{prefix}{i}{postfix}.sdf'))[0]
            # breakpoint()
            res = gnina_score(pdb_path, sample, fix_pose=fix_pose)
            # breakpoint()
            print(f'{i} len: {sample.GetNumAtoms()}: {res.best.affinity}')
        except:
            print(f'{i} len: {sample.GetNumAtoms()}: {Chem.MolToSmiles(sample)}')

# # ---------- Example usage ----------
# if __name__ == "__main__":
#     from rdkit import Chem
#     smi = "c1ccccc1O"  # phenol
#     mol = Chem.MolFromSmiles(smi)
#     mol = Chem.AddHs(mol)
#     # If you want a fixed pose, generate 3D coords; otherwise gnina will search poses itself.
#     # from rdkit.Chem import AllChem
#     # AllChem.EmbedMolecule(mol, AllChem.ETKDG()); AllChem.UFFOptimizeMolecule(mol)

#     result = gnina_score("/path/to/protein.pdb", mol, fix_pose=False)
#     print("Command:", result.cmd)
#     print("Best pose:", result.best)
#     # Dump as JSON
#     print(json.dumps([p.__dict__ for p in result.poses], indent=2))
