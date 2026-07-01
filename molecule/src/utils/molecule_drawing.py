import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


def molecule_to_topology_png(
    mol: Chem.Mol,
    *,
    width: int = 600,
    height: int = 600,
    add_atom_indices: bool = True,
) -> bytes:
    """Render a 2D topology PNG for an RDKit molecule."""
    mol_to_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_to_draw)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = add_atom_indices

    drawer.DrawMolecule(mol_to_draw)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def save_molecule_topology_image(
    mol: Chem.Mol,
    output_path: Path,
    *,
    width: int = 600,
    height: int = 600,
    add_atom_indices: bool = True,
) -> bool:
    """Save a 2D topology PNG for an RDKit molecule."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            molecule_to_topology_png(
                mol,
                width=width,
                height=height,
                add_atom_indices=add_atom_indices,
            )
        )
        return True
    except Exception as exc:
        logger.warning("Failed to save molecule topology image to %s: %s", output_path, exc)
        return False


def save_sdf_topology_image(
    sdf_path: Path,
    output_path: Path | None = None,
    *,
    sanitize: bool = False,
    remove_hs: bool = False,
    width: int = 600,
    height: int = 600,
    add_atom_indices: bool = True,
) -> bool:
    """Read the first molecule from an SDF file and save its 2D topology PNG."""
    output_path = output_path or sdf_path.with_suffix(".png")
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize, removeHs=remove_hs)
        mol = next((sample for sample in supplier if sample is not None), None)
    except Exception as exc:
        logger.warning("Failed to read SDF file for topology image %s: %s", sdf_path, exc)
        return False

    if mol is None:
        logger.warning("No molecule found in SDF file for topology image: %s", sdf_path)
        return False

    return save_molecule_topology_image(
        mol,
        output_path,
        width=width,
        height=height,
        add_atom_indices=add_atom_indices,
    )
