import os
import shutil
import argparse
import torch
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from tqdm import tqdm
from utils.metrics.qvina import qvina_score_from_mol
from utils.utils import sandbox_import_dir, replace_mol_topology_by_fragment

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--i_prefix', type=str, default='')
    parser.add_argument('--i_postfix', type=str, default='')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--fix_pose', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--csv_save_path', type=str, default=None)
    args = parser.parse_args(argv)
    
    ligand_dir = args.ligand_dir
    project_root = get_project_root()
    data_root = os.path.join(project_root, 'data', 'crossdock', 'raw_sbdd')
    protein_dir = os.path.join(project_root, 'src', 'pretrained_models', 'DiffSBDD', 'dataset', 'crossdock', 'crossdocked_pocket10')
    max_len = args.num_samples
    os.makedirs(args.save_dir, exist_ok=True)

    output_dict = {}
    task_list = [14, 15, 27, 38, 43, 60, 69, 77, 78]
    for i in task_list:
        data_path = os.path.join(data_root, f'{i}.pt')
        if not os.path.exists(data_path):
            continue
        with sandbox_import_dir(os.path.join(project_root, 'baseline', 'Delete')):
            data = torch.load(data_path)
        scaffold = data.scaffold
        ref_length = data.mol.GetNumAtoms()
        protein_filename = data.protein_filename
        protein_path = os.path.join(protein_dir, protein_filename)
        eval_res_list = []
        print(f'[{i}] protein_path: {protein_path}')
        for j in range(max_len):
            gen_ligand_path = os.path.join(ligand_dir, f'{args.i_prefix}{i}{args.i_postfix}', f'{args.prefix}{j}{args.postfix}.sdf')
            print(f'[{i}] gen_ligand_path: {gen_ligand_path}')
            if not os.path.exists(gen_ligand_path):
                eval_res_list.append(0)
                continue
            print(f'[{i}] gen_ligand_path exists')
            try:
                gen_ligand = Chem.SDMolSupplier(gen_ligand_path, sanitize=False)[0]
                if args.replace:  
                    gen_ligand = replace_mol_topology_by_fragment(gen_ligand, scaffold, list(range(scaffold.GetNumAtoms())))
                eval_res = qvina_score_from_mol(protein_path, gen_ligand)
            except:
                print(f'docking failed: {gen_ligand_path}')
                score = 0
                continue
            if eval_res.best is None:
                print(f'docking failed: {gen_ligand_path}')
                score = 0
            else:
                score = eval_res.best.affinity
            eval_res_list.append(score)
        eval_res_list = np.array(eval_res_list)
        eval_res_list_sorted = eval_res_list[eval_res_list.argsort()]
        print(f'{i} [len {ref_length}] | {eval_res_list_sorted}')

        output_dict[i] = {'protein_filename': protein_filename, 'ref_length': ref_length, 'scores': eval_res_list_sorted}
        score_save_path = os.path.join(args.save_dir, f'{i}.pt')
        torch.save(output_dict[i], score_save_path)

    df = pd.DataFrame.from_dict(output_dict)
    df.transpose().to_csv(args.csv_save_path)
    torch.save(output_dict, os.path.join(args.save_dir, f'all_scores.pt'))