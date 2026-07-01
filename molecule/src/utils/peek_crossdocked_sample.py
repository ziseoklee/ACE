from pathlib import Path
from typing import Any

import click
import torch
from rdkit.Chem import Mol

from utils.molecule_drawing import molecule_to_topology_png

DEFAULT_PROCESSED_DIR = Path("data/crossdocked/processed")
DEFAULT_OUTPUT_DIR = Path("tmp/crossdocked_peek")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("sample")
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_PROCESSED_DIR,
    show_default=True,
    help="Directory containing CrossDocked2020 processed .pt files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory for rendered PNG files.",
)
def main(sample: str, processed_dir: Path, output_dir: Path) -> None:
    """Peek at one CrossDocked2020 processed .pt sample and render topology images."""
    pt_path = resolve_pt_path(sample, processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = torch.load(pt_path, weights_only=False)
    fragment = get_mol(data, "scaffold")
    ligand = get_ligand_mol(data)

    fragment_png = output_dir / f"{pt_path.stem}_fragment.png"
    ligand_png = output_dir / f"{pt_path.stem}_ligand.png"

    fragment_png.write_bytes(molecule_to_topology_png(fragment))
    ligand_png.write_bytes(molecule_to_topology_png(ligand))

    click.echo(f"pt_file: {pt_path}")
    click.echo(f"protein_pocket_filename: {data.get('protein_filename', '<missing>')}")
    click.echo(f"scaffold_smiles: {data.get('scaffold_smiles', '<missing>')}")
    click.echo(f"fragment_atoms/bonds: {fragment.GetNumAtoms()}/{fragment.GetNumBonds()}")
    click.echo(f"ligand_atoms/bonds: {ligand.GetNumAtoms()}/{ligand.GetNumBonds()}")
    click.echo(f"fragment_topology_png: {fragment_png}")
    click.echo(f"ligand_topology_png: {ligand_png}")


def resolve_pt_path(sample: str, processed_dir: Path) -> Path:
    candidate = Path(sample)
    if candidate.exists():
        return candidate

    if candidate.suffix == ".pt":
        pt_path = processed_dir / candidate.name
    else:
        pt_path = processed_dir / f"{sample}.pt"

    if not pt_path.exists():
        raise FileNotFoundError(f"CrossDocked2020 processed sample does not exist: {pt_path}")
    return pt_path


def get_ligand_mol(data: dict[str, Any]) -> Mol:
    if isinstance(data.get("mol"), Mol):
        return data["mol"]
    if isinstance(data.get("ligand_mol"), Mol):
        return data["ligand_mol"]
    raise KeyError("Could not find ligand molecule under `mol` or `ligand_mol`.")


def get_mol(data: dict[str, Any], key: str) -> Mol:
    mol = data.get(key)
    if not isinstance(mol, Mol):
        raise KeyError(f"Could not find RDKit molecule under `{key}`.")
    return mol


if __name__ == "__main__":
    main()
