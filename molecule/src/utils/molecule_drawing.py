from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


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


def sdf_to_first_mol(
    sdf_path: Path,
    *,
    sanitize: bool = False,
    remove_hs: bool = False,
) -> Chem.Mol | None:
    """Read the first molecule from an SDF file."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize, removeHs=remove_hs)
    return next((sample for sample in supplier if sample is not None), None)
