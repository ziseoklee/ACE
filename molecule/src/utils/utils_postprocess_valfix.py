import logging
import shutil
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

from utils.enforce_frag import reconstruct_molecule_with_scaffold

logger = logging.getLogger(__name__)


def postprocess_valfix(
    data_root: Path,
    output_ligand_dir: Path,
    num_samples: int,
):

    protein_dir = data_root / "crossdocked_pocket10"
    processed_data_root = data_root / "processed"

    for i in tqdm(range(100)):
        data_path = processed_data_root / f"{i}.pt"
        if not data_path.exists():
            continue

        data = torch.load(data_path, weights_only=False)
        scaffold = data["scaffold"]
        ref_length = data["mol"].GetNumAtoms()
        if ref_length > 29:
            continue

        protein_filename = data["protein_filename"]
        protein_path = protein_dir / protein_filename
        logger.info(f"[{i}] protein_path: {protein_path}")

        for j in range(num_samples):
            gen_ligand_path = output_ligand_dir / f"{i}" / f"{j}.sdf"
            logger.info(f"[{i}] gen_ligand_path: {gen_ligand_path}")
            if not gen_ligand_path.exists():
                continue
            logger.info(f"[{i}] gen_ligand_path exists")
            save_path = gen_ligand_path.with_name(gen_ligand_path.stem + "_recon.sdf")
            gen_ligand = Chem.SDMolSupplier(str(gen_ligand_path), sanitize=False)[0]
            if gen_ligand is None:
                shutil.copyfile(gen_ligand_path, save_path)
                continue
            recon_ligand = reconstruct_molecule_with_scaffold(gen_ligand, scaffold)
            with Chem.SDWriter(str(save_path)) as writer:
                writer.write(recon_ligand)
            assert len(Chem.GetMolFrags(recon_ligand)) == 1, f"{Chem.MolToSmiles(recon_ligand)}"

    logger.info("Done!")
