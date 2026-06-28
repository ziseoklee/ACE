"""
Apply runtime compatibility fixes to vendored pretrained expert sources.

The patches here are intentionally small, exact, and idempotent. They keep the
upstream expert repositories usable in the current ACE runtime without requiring
forks or permanent submodule commits.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_ROOT = PROJECT_ROOT / "src" / "pretrained_models"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text()


def write_text_if_changed(path: Path, old_text: str, new_text: str) -> None:
    if new_text != old_text:
        path.write_text(new_text)


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = read_text(path)
    if new in text:
        print(f"  [skip] {label}")
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected text for {label} in {path}")
    write_text_if_changed(path, text, text.replace(old, new, 1))
    print(f"  [ok] {label}")


def replace_all(path: Path, old: str, new: str, label: str) -> None:
    text = read_text(path)
    if old not in text:
        print(f"  [skip] {label}")
        return
    write_text_if_changed(path, text, text.replace(old, new))
    print(f"  [ok] {label}")


def ensure_line(path: Path, line: str, label: str) -> None:
    text = path.read_text() if path.exists() else ""
    lines = text.splitlines()
    if line in lines:
        print(f"  [skip] {label}")
        return
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line + "\n")
    print(f"  [ok] {label}")


def patch_diffsbdd() -> None:
    root = PRETRAINED_ROOT / "DiffSBDD"

    ensure_line(root / ".gitignore", "__pycache__/", "DiffSBDD .gitignore ignores __pycache__")

    lightning_modules = root / "lightning_modules.py"
    replace_all(
        lightning_modules,
        "from Bio.PDB.Polypeptide import three_to_one",
        "from Bio.Data.IUPACData import protein_letters_3to1",
        "DiffSBDD Biopython import",
    )
    replace_all(
        lightning_modules,
        "three_to_one(res.get_resname())",
        "protein_letters_3to1[res.get_resname().title()]",
        "DiffSBDD residue name conversion",
    )

    utils = root / "utils.py"
    old_get_pocket_header = (
        'def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):\n\n    if ligand.endswith(".sdf"):\n'
    )
    new_get_pocket_header = (
        "def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):\n"
        "    if not isinstance(ligand, str):\n"
        "        ligand_coords = torch.from_numpy(ligand.GetConformer().GetPositions()).float()\n"
        "        resi = None\n"
        '    elif ligand.endswith(".sdf"):\n'
    )
    replace_once(
        utils,
        old_get_pocket_header,
        new_get_pocket_header,
        "DiffSBDD RDKit Mol ligand pocket support",
    )

    molecule_builder = root / "analysis" / "molecule_builder.py"
    replace_once(
        molecule_builder,
        "import openbabel\n",
        "from openbabel import openbabel\n",
        "DiffSBDD Open Babel import",
    )


def patch_geodiff() -> None:
    root = PRETRAINED_ROOT / "GeoDiff"
    gin_encoder = root / "models" / "encoder" / "gin.py"
    replace_all(
        gin_encoder,
        "hidden += conv_input",
        "hidden = hidden + conv_input",
        "GeoDiff non-inplace residual update",
    )


def main() -> None:
    print("Applying runtime compatibility patches...")
    patch_diffsbdd()
    patch_geodiff()
    print("Runtime compatibility patches complete.")


if __name__ == "__main__":
    main()
