import argparse
import logging
import os
import shutil

import torch
from rdkit import Chem
from tqdm import tqdm

from utils.enforce_frag import reconstruct_molecule_with_scaffold

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/crossdocked")
    parser.add_argument("--ligand_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--i_prefix", type=str, default="")
    parser.add_argument("--i_postfix", type=str, default="")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--postfix", type=str, default="")
    args = parser.parse_args()

    ligand_dir = args.ligand_dir
    data_root = args.data_root
    protein_dir = os.path.join(data_root, "crossdocked_pocket10")
    data_root = os.path.join(data_root, "processed")
    max_len = args.num_samples

    task_list = list(range(100))
    for i in tqdm(task_list):
        data_path = os.path.join(data_root, f"{i}.pt")
        if not os.path.exists(data_path):
            continue
        data = torch.load(data_path)
        scaffold = data["scaffold"]
        ref_length = data["mol"].GetNumAtoms()
        if ref_length > 29:
            continue
        protein_filename = data["protein_filename"]
        protein_path = os.path.join(protein_dir, protein_filename)
        logger.info(f"[{i}] protein_path: {protein_path}")

        for j in range(max_len):
            gen_ligand_path = os.path.join(
                ligand_dir,
                f"{args.i_prefix}{i}{args.i_postfix}",
                f"{args.prefix}{j}{args.postfix}.sdf",
            )
            logger.info(f"[{i}] gen_ligand_path: {gen_ligand_path}")
            if not os.path.exists(gen_ligand_path):
                continue
            logger.info(f"[{i}] gen_ligand_path exists")
            save_path = gen_ligand_path.replace(".sdf", "_recon.sdf")
            gen_ligand = Chem.SDMolSupplier(gen_ligand_path, sanitize=False)[0]
            if gen_ligand is None:
                shutil.copyfile(gen_ligand_path, save_path)
                continue
            recon_ligand = reconstruct_molecule_with_scaffold(gen_ligand, scaffold)
            with Chem.SDWriter(save_path) as writer:
                writer.write(recon_ligand)
            assert len(Chem.GetMolFrags(recon_ligand)) == 1, f"{Chem.MolToSmiles(recon_ligand)}"

    logger.info("Done!")


if __name__ == "__main__":
    main()
