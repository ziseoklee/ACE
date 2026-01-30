import importlib
import sys
import pickle as pkl
import torch
import rdkit
import numpy as np
import py3Dmol
import torch_geometric
import os
import functools
import pandas as pd
import shutil
import argparse
from torch_geometric.data import Data
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from utils.utils import load_diffsbdd, load_geodiff, load_e3_diffusion, sandbox_import_dir, replace_mol_topology_by_fragment, get_project_root
from utils.metrics.gnina import gnina_score


from src.hcg import HcgSingleConditionMaskExpanded, generate_samples_weighted, ProbabilityPathHcgWraper
from src.probability_path import ProbabilityPath, ConcatenatedProbabilityPath, GeodiffSchedule, EdmSchedule, DiffSBDDSchedule
from src.distributions import FixedPointDistribution, sample_product_masked_diag

with sandbox_import_dir(os.path.join(get_project_root(), 'src', 'pretrained_models', 'DiffSBDD')):
    from export import prepare_data as prepare_data_diffsbdd
    from export import score_function as score_function_diffsbdd
    from export import interleave_fn as interleave_fn_diffsbdd
    from export import postprocess_fn as postprocess_fn_diffsbdd
    from export import build_molecule, process_molecule
with sandbox_import_dir(os.path.join(get_project_root(), 'src', 'pretrained_models', 'GeoDiff')):
    from export import prepare_data as prepare_data_geodiff
    from export import score_function as score_function_geodiff
    from export import interleave_fn as interleave_fn_geodiff
    from export import postprocess_fn as postprocess_fn_geodiff
with sandbox_import_dir(os.path.join(get_project_root(), 'src', 'pretrained_models', 'e3_diffusion_for_molecules')):
    from export import prepare_data as prepare_data_edm
    from export import score_function as score_function_edm
    from export import interleave_fn as interleave_fn_edm
    from export import postprocess_fn as postprocess_fn_edm
    from export import construct_rdkit_molecules
    from export import encode_xh, decode_h



def vis_mol(pdb_block, protein_pdb_block=None):
    view = py3Dmol.view()
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'model': 0}, {'stick':{'colorscheme':'greenCarbon'}})
    if protein_pdb_block is not None:
        view.addModel(protein_pdb_block, 'pdb')
        view.addSurface(py3Dmol.VDW, {'opacity':0.85, 'color':'white'}, \
        {'not':{'or':[{'resn':'UH7'}, {'resn':'DMS'}]}})
    view.zoomTo({'model': 0})
    view.show()
    

def gnina_score(pdb_path, rdkit_mol, fix_pose=False):
    tmp_ligand_path = f'/home/minyeong/HCG/tmp_ligand.pdb'
    Chem.MolToPDBFile(rdkit_mol, tmp_ligand_path)
    if not fix_pose:
        cmd = f'gnina -r {pdb_path} -l {tmp_ligand_path} -o tmp.pdb --autobox_ligand {tmp_ligand_path} --seed 42 --cpu 1'
    else:
        cmd = f'gnina --score_only -r {pdb_path} -l {tmp_ligand_path} -o tmp.pdb --seed 42 --cpu 1'
    os.system(cmd)
    
    # return score
def make_data_component_mask_extend(num_atoms_subset, num_atoms, type='sbdd', device='cpu'):
    # space allocation: num_atoms_subset x (coord, sbdd_atom_types, include_charge) + (num_atoms - num_atoms_subset) x (coord, edm_atom_types), we include 'H' into 'others' type
    # edm_atoms = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    # sbdd_atoms = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
    assert type == 'sbdd'

    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_geodiff_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_geodiff_part1[:, :3] = True
    mask_geodiff_part1 = mask_geodiff_part1.flatten()
    mask_geodiff_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool().flatten()
    mask_geodiff = torch.cat([mask_geodiff_part1, mask_geodiff_part2], dim=0)
    
    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_geodiff_h_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_geodiff_h_part1[:, 3:] = True
    mask_geodiff_h_part1 = mask_geodiff_h_part1.flatten()
    mask_geodiff_h_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool().flatten()
    mask_geodiff_h = torch.cat([mask_geodiff_h_part1, mask_geodiff_h_part2], dim=0)

    mask_subset = mask_geodiff + mask_geodiff_h

    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_edm_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_edm_part1[:, :3] = True
    mask_edm_part1[:, [3 + edm2sbdd_atoms[i] for i in edm2sbdd_atoms]] = True
    mask_edm_part1[:, -1] = True
    mask_edm_part1 = mask_edm_part1.flatten()

    mask_edm_h_part1 = ~mask_edm_part1

    mask_edm_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool()
    mask_edm_part2[:, :] = True
    mask_edm_part2 = mask_edm_part2.flatten()

    mask_edm = torch.cat([mask_edm_part1, mask_edm_part2], dim=0)

    mask_sbdd_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_sbdd_part1[:, :13] = True
    mask_sbdd_part1 = mask_sbdd_part1.flatten()
    mask_sbdd_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool()
    mask_sbdd_part2[:, :13] = True
    mask_sbdd_part2 = mask_sbdd_part2.flatten()
    mask_sbdd = torch.cat([mask_sbdd_part1, mask_sbdd_part2], dim=0)
    
    mask_sbdd_hpad_part1 = torch.zeros(num_atoms_subset, 15).bool()
    mask_sbdd_hpad_part1[:, 13:] = True
    mask_sbdd_hpad_part1 = mask_sbdd_hpad_part1.flatten()
    mask_sbdd_hpad_part2 = torch.zeros(num_atoms - num_atoms_subset, 15).bool()
    mask_sbdd_hpad_part2[:, 13:] = True
    mask_sbdd_hpad_part2 = mask_sbdd_hpad_part2.flatten()
    mask_sbdd_hpad = torch.cat([mask_sbdd_hpad_part1, mask_sbdd_hpad_part2], dim=0)
    
    mask = mask_sbdd + mask_sbdd_hpad

    mask_edm2 = torch.zeros(num_atoms, 15).bool()
    mask_edm2[:, :3] = True
    mask_edm2[:, [3 + edm2sbdd_atoms[i] for i in edm2sbdd_atoms]] = True
    mask_edm2[:, -1] = True
    mask_edm2 = mask_edm2.flatten()
    mask_edm2_h = ~mask_edm2
    return mask_geodiff_part1.to(device), mask_geodiff_h_part1.to(device), mask_edm_part1.to(device), mask_edm_h_part1.to(device), mask_subset.to(device), mask_sbdd.to(device), mask_sbdd_hpad.to(device), mask.to(device), mask_edm2.to(device), mask_edm2_h.to(device)


def hcg_flatten_space(pdb_path, fragment, ref_ligand_path, sbdd, edm, geodiff, args_edm, args_geodiff, num_samples=10, num_nodes=10, n_steps=500, device='cpu', inverse_temperature=1.0, use_bump=False):
    ### GeoDiff Probability Path
    scheduler_geodiff = GeodiffSchedule()
    scheduler_edm = EdmSchedule()
    scheduler_sbdd = DiffSBDDSchedule()

    N = num_nodes
    n = fragment.GetNumAtoms()
    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_geodiff_part1, mask_geodiff_h_part1, mask_edm_part1, mask_edm_h_part1, mask_subset, mask_sbdd, mask_sbdd_hpad, mask, mask_edm2, mask_edm2_h = make_data_component_mask_extend(n, N, type='sbdd', device=device)

    fragment_center = torch.tensor(fragment.GetConformer(0).GetPositions().mean(axis=0)).to(device)

    len_x = fragment.GetNumAtoms()

    
    N = num_nodes
    n = fragment.GetNumAtoms()
    edm2sbdd_atoms = {0: 10, 1: 0, 2: 1, 3: 2, 4: 9}
    mask_geodiff_part1, mask_geodiff_h_part1, mask_edm_part1, mask_edm_h_part1, mask_subset, mask_sbdd, mask_sbdd_hpad, mask, mask_edm2, mask_edm2_h = make_data_component_mask_extend(n, N, type='sbdd', device=device)

    prepared_data_geodiff = prepare_data_geodiff(args_geodiff, geodiff, fragment, num_samples=num_samples, device=device)
    score_fn = functools.partial(score_function_geodiff, prepared_data=prepared_data_geodiff)
    q_geodiff = ProbabilityPath(scheduler_geodiff, score_fn)

    _, _h = encode_xh(args_edm, edm, fragment)
    h_int = _h[:, :-1].argmax(dim=-1)
    h_int = torch.tensor([edm2sbdd_atoms[v.item()] for v in h_int]).to(device)
    # print(f'h_int: {h_int}')
    h = torch.zeros(n, 12).to(device=device)
    h[:, list(edm2sbdd_atoms.values()) + [-1]] = _h.to(device=device)
    h_dist = FixedPointDistribution(h.flatten(), device=device)
    h_score = h_dist.export_score_function(scheduler_geodiff)
    q_h = ProbabilityPath(scheduler_geodiff, h_score)

    q_geodiff_pad = ConcatenatedProbabilityPath([q_geodiff, q_h], [mask_geodiff_part1, mask_geodiff_h_part1])

    prepared_data_edm = prepare_data_edm(edm, num_samples, n, device=device)
    score_fn_edm = functools.partial(score_function_edm, prepared_data=prepared_data_edm)
    q_edm = ProbabilityPath(scheduler_edm, score_fn_edm)

    h = torch.zeros(n, 6).to(device=device)
    h_dist = FixedPointDistribution(h.flatten(), device=device)
    h_score = h_dist.export_score_function(scheduler_edm)
    q_h = ProbabilityPath(scheduler_edm, h_score)

    q_edm_pad = ConcatenatedProbabilityPath([q_edm, q_h], [mask_edm_part1, mask_edm_h_part1])

    prepared_data_edm2 = prepare_data_edm(edm, num_samples, N, device=device)
    score_fn_edm2 = functools.partial(score_function_edm, prepared_data=prepared_data_edm2)
    q_edm2 = ProbabilityPath(scheduler_edm, score_fn_edm2)

    h2 = torch.zeros(N, 6).to(device=device)
    h2_dist = FixedPointDistribution(h2.flatten(), device=device)
    h2_score = h2_dist.export_score_function(scheduler_edm)
    q_h2 = ProbabilityPath(scheduler_edm, h2_score)

    q_edm_pad2 = ConcatenatedProbabilityPath([q_edm2, q_h2], [mask_edm2, mask_edm2_h])

    prepared_data_sbdd = prepare_data_diffsbdd(sbdd, pdb_path, ref_ligand_path, num_samples, N, device=device)
    score_fn_sbdd = functools.partial(score_function_diffsbdd, prepared_data=prepared_data_sbdd)
    q_sbdd = ProbabilityPath(scheduler_sbdd, score_fn_sbdd)

    h = torch.zeros(N, 2).to(device=device)
    h_dist = FixedPointDistribution(h.flatten(), device=device)
    h_score = h_dist.export_score_function(scheduler_sbdd)
    q_h = ProbabilityPath(scheduler_sbdd, h_score)

    q_sbdd_pad = ConcatenatedProbabilityPath([q_sbdd, q_h], [mask_sbdd, mask_sbdd_hpad])

    prior_sbdd = prepared_data_sbdd['z']
    prior = torch.randn(num_samples, *mask.shape).to(prior_sbdd.device)
    prior[:, mask_sbdd] = prior_sbdd

    # Compute log probability of gaussian prior
    from torch.distributions import Normal
    standard_normal_dist = Normal(loc=0.0, scale=1.0)
    log_probs = standard_normal_dist.log_prob(prior)
    log_probs = log_probs.sum(dim=-1).unsqueeze(1).repeat(1, 4).unsqueeze(2)

    final_dim = len(mask)

    interleave_fn_sbdd = functools.partial(interleave_fn_diffsbdd, prepared_data=prepared_data_sbdd, mask=mask_sbdd)
    if not use_bump:
        exponent_list = [lambda t: 0 * t + inverse_temperature, lambda t: 0 * t - inverse_temperature, lambda t:  0 * t + inverse_temperature, lambda t: 0 * t - (inverse_temperature - 1)]
    else:
        exponent_list = [lambda t: 0 * t + inverse_temperature, lambda t:  - inverse_temperature ** t, lambda t:  30 * t * (1 - t) + inverse_temperature, lambda t:  0 * t - (inverse_temperature - 1)]

    hcg = HcgSingleConditionMaskExpanded(scheduler_geodiff, [q_geodiff_pad, q_edm_pad, q_sbdd_pad, q_edm_pad2],  mask_list=[mask_subset, mask_subset, mask, mask], exponent_list=exponent_list, dim=final_dim, gamma=2.5, use_bump=use_bump)
    
    samples, _, _ = generate_samples_weighted(hcg, t_span=(0, 1), resampling_interval=10, num_integration_steps=500, samples=prior, logq_tensor=log_probs, interleave_fn=interleave_fn_sbdd)

    samples = postprocess_fn_diffsbdd(samples.to(device=device), prepared_data=prepared_data_sbdd, mask=mask_sbdd, frag_atom_type=h_int)

    replaced_samples = []
    for sample in samples:
        try:
            sample = replace_mol_topology_by_fragment(sample, fragment, list(range(fragment.GetNumAtoms())))
            replaced_samples.append(sample)
        except:
            continue

    valid_samples = replaced_samples

    return valid_samples

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='data/crossdock/raw')
    parser.add_argument('--save_dir', type=str, default='result_inference/crossdock_1.3_bump')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_bump', action='store_true')
    parser.add_argument('--inverse_temperature', type=float, default=1.0)
    args = parser.parse_args(argv)
    num_trials = args.num_trials
    data_root = args.data_root
    _save_dir = args.save_dir
    device = args.device
    use_bump = args.use_bump

    sbdd  = load_diffsbdd(device=device)
    args_edm, edm = load_e3_diffusion(device=device)
    args_geodiff, geodiff = load_geodiff(device=device)

    task_list = [14, 15, 27, 38, 43, 60, 69, 77, 78]

    for i in task_list:
        print(f'Start {i}')
        data_path = os.path.join(data_root, f'{i}.pt')
        if not os.path.exists(data_path):
            continue
        with sandbox_import_dir(os.path.join(get_project_root(), 'baseline', 'Delete')):
            data = torch.load(data_path)
        protein_dir = os.path.join(get_project_root(), 'src', 'pretrained_models', 'DiffSBDD', 'dataset', 'crossdock', 'crossdocked_pocket10')
        pdb_path = os.path.join(protein_dir, f'{data.protein_filename}')
        ligand_path = pdb_path.replace('_pocket10.pdb', '.sdf')
        ligand, fragment = data.mol, data.scaffold
        N = ligand.GetNumAtoms()

        valid_samples_list = []
        for j in range(num_trials):
            samples = hcg_flatten_space(pdb_path, fragment, ligand, sbdd, edm, geodiff, args_edm, args_geodiff, num_samples=10, num_nodes=N, n_steps=500, device=device, inverse_temperature=args.inverse_temperature, use_bump=args.use_bump)
            valid_samples = samples
            valid_samples_list.extend(valid_samples)

        output_samples = np.random.permutation(valid_samples_list)
        save_dir = os.path.join(_save_dir, f'{i}')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for j, sample in enumerate(output_samples):
            try:
                writer = Chem.SDWriter(save_dir + f'/{j}.sdf')
                writer.write(sample)
                writer.close()
            except:
                continue


if __name__ == '__main__':
    main(sys.argv[1:])