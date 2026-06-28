"""
Prepare pretrained expert sources as editable, namespaced dependencies.

This script copies packaging metadata from assets into the vendored expert
directories and rewrites legacy absolute imports to namespaced imports. It is
idempotent: running it multiple times should not duplicate package prefixes.
"""

import re
import shutil
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets" / "pretrained_packaging"
PRETRAINED_ROOT = PROJECT_ROOT / "src" / "pretrained_models"


RegexReplacement = tuple[str, str]


def copy_pyproject(asset_dir: str, expert_dir: str) -> None:
    src = ASSET_ROOT / asset_dir / "pyproject.toml"
    dst = PRETRAINED_ROOT / expert_dir / "pyproject.toml"
    if not src.exists():
        raise FileNotFoundError(src)
    if not dst.parent.exists():
        raise FileNotFoundError(dst.parent)
    shutil.copy2(src, dst)
    print(f"  [ok] copied {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(PROJECT_ROOT)}")


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def apply_regex_replacements(root: Path, replacements: list[RegexReplacement], label: str) -> None:
    changed_files = 0
    for path in iter_python_files(root):
        old_text = path.read_text()
        new_text = old_text
        for pattern, repl in replacements:
            new_text = re.sub(pattern, repl, new_text, flags=re.MULTILINE)
        if new_text != old_text:
            path.write_text(new_text)
            changed_files += 1
    print(f"  [ok] {label}: updated {changed_files} files")


def patch_e3_imports() -> None:
    root = PRETRAINED_ROOT / "e3_diffusion_for_molecules"
    replacements = [
        (r"^from configs\.", "from e3_diffusion_for_molecules.configs."),
        (r"^import configs\.", "import e3_diffusion_for_molecules.configs."),
        (r"^from qm9\.", "from e3_diffusion_for_molecules.qm9."),
        (r"^import qm9\.", "import e3_diffusion_for_molecules.qm9."),
        (r"^from qm9 import ", "from e3_diffusion_for_molecules.qm9 import "),
        (r"^import qm9$", "import e3_diffusion_for_molecules.qm9 as qm9"),
        (r"^from equivariant_diffusion\.", "from e3_diffusion_for_molecules.equivariant_diffusion."),
        (r"^from equivariant_diffusion import ", "from e3_diffusion_for_molecules.equivariant_diffusion import "),
        (r"^import equivariant_diffusion\.", "import e3_diffusion_for_molecules.equivariant_diffusion."),
        (r"^from egnn\.", "from e3_diffusion_for_molecules.egnn."),
        (r"^from egnn import ", "from e3_diffusion_for_molecules.egnn import "),
        (r"^import utils$", "from e3_diffusion_for_molecules import utils"),
    ]
    apply_regex_replacements(root, replacements, "e3_diffusion_for_molecules namespaced imports")


def patch_diffsbdd_imports() -> None:
    root = PRETRAINED_ROOT / "DiffSBDD"
    replacements = [
        (r"^from constants import ", "from diffsbdd.constants import "),
        (r"^import constants$", "from diffsbdd import constants"),
        (r"^from analysis\.", "from diffsbdd.analysis."),
        (r"^from analysis import ", "from diffsbdd.analysis import "),
        (r"^from equivariant_diffusion\.", "from diffsbdd.equivariant_diffusion."),
        (r"^from equivariant_diffusion import ", "from diffsbdd.equivariant_diffusion import "),
        (r"^from dataset import ", "from diffsbdd.dataset import "),
        (r"^from geometry_utils import ", "from diffsbdd.geometry_utils import "),
        (r"^import utils$", "from diffsbdd import utils"),
    ]
    apply_regex_replacements(root, replacements, "DiffSBDD namespaced imports")


def patch_geodiff_imports() -> None:
    root = PRETRAINED_ROOT / "GeoDiff"
    replacements = [
        (r"^from models\.", "from geodiff.models."),
        (r"^from models import ", "from geodiff.models import "),
        (r"^import models\.", "import geodiff.models."),
        (r"^from utils\.", "from geodiff.utils."),
        (r"^from utils import ", "from geodiff.utils import "),
        (r"^import utils\.", "import geodiff.utils."),
    ]
    apply_regex_replacements(root, replacements, "GeoDiff namespaced imports")


def main() -> None:
    print("Applying pretrained dependency packaging patches...")
    copy_pyproject("e3_diffusion_for_molecules", "e3_diffusion_for_molecules")
    copy_pyproject("diffsbdd", "DiffSBDD")
    copy_pyproject("geodiff", "GeoDiff")
    patch_e3_imports()
    patch_diffsbdd_imports()
    patch_geodiff_imports()
    print("Pretrained dependency packaging patches complete.")


if __name__ == "__main__":
    main()
