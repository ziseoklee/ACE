import logging
from pathlib import Path

from rdkit import Chem

from postprocessing.fragment_enforcement import reconstruct_molecule_with_scaffold

logger = logging.getLogger(__name__)


def postprocess_valfix(
    protein_pocket_pdb_path: Path,
    fragment_sdf_path: Path,
    ref_ligand_sdf_path: Path,
    output_ligand_dir: Path,
    num_samples: int,
):
    logger.info(f"protein_pocket_pdb_path: {protein_pocket_pdb_path}")
    logger.info(f"fragment_sdf_path: {fragment_sdf_path}")
    logger.info(f"ref_ligand_sdf_path: {ref_ligand_sdf_path}")
    logger.info(f"output_ligand_dir: {output_ligand_dir}")

    fragment_mol: Chem.Mol = Chem.SDMolSupplier(str(fragment_sdf_path), sanitize=False)[0]
    ref_ligand_mol: Chem.Mol = Chem.SDMolSupplier(str(ref_ligand_sdf_path), sanitize=False)[0]
    if num_ligand_atoms := ref_ligand_mol.GetNumAtoms() > 29:
        logger.warning(
            f"29 atoms is the maximum supported number of atoms for the ligand in EDM training. However, the ligand in input data has {num_ligand_atoms} atoms, which exceeds the limit. Please note that results may be unreliable."
        )

    # For each generated ligand, reconstruct it with the fragment and save it to output_ligand_dir.
    for j in range(num_samples):
        output_ligand_sdf_path = output_ligand_dir / f"{j}.sdf"
        logger.info(f"gen_ligand_path: {output_ligand_sdf_path}")
        if not output_ligand_sdf_path.exists():
            logger.warning(f"Generated ligand path does not exist: {output_ligand_sdf_path}")
            continue
        logger.info(f"[{j}] output_ligand_sdf_path exists")
        output_ligand_mol: Chem.Mol = Chem.SDMolSupplier(str(output_ligand_sdf_path), sanitize=False)[0]
        if output_ligand_mol is None:
            logger.warning(f"Failed to read generated ligand from path: {output_ligand_sdf_path}")
            continue

        recon_ligand_mol = reconstruct_molecule_with_scaffold(output_ligand_mol, fragment_mol)
        if recon_ligand_mol is None:
            logger.warning(f"Failed to reconstruct molecule for generated ligand at path: {output_ligand_sdf_path}")
            continue

        recon_ligand_save_path = output_ligand_dir / f"{j}_recon.sdf"
        with Chem.SDWriter(str(recon_ligand_save_path)) as writer:
            writer.write(recon_ligand_mol)
        assert len(Chem.GetMolFrags(recon_ligand_mol)) == 1, f"{Chem.MolToSmiles(recon_ligand_mol)}"
        logger.info(f"Reconstructed ligand saved to: {recon_ligand_save_path}")
