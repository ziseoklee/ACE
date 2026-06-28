from pathlib import Path

from rdkit import Chem

PRETRAINED_MODEL_DIR = Path(__file__).parents[1] / "pretrained_models"


def load_diffsbdd(device="cpu"):
    """Instantiate the DiffSBDD model and load checkpoint weights."""
    ckpt = PRETRAINED_MODEL_DIR / "DiffSBDD" / "checkpoints" / "crossdocked_fullatom_cond.ckpt"

    from pretrained_models.export_diffsbdd import export_diffsbdd

    model = export_diffsbdd(ckpt, device=device)
    model.to(device)
    return model


def load_geodiff(device="cpu"):
    """Instantiate the GeoDiff model and load checkpoint weights."""
    ckpt = PRETRAINED_MODEL_DIR / "GeoDiff" / "log" / "model" / "checkpoints" / "qm9_default.pt"

    from pretrained_models.export_geodiff import export_geodiff

    config, model = export_geodiff(ckpt, device=device)
    model.to(device)
    return config, model


def load_edm(device="cpu"):
    """Instantiate the EDM model and load checkpoint weights."""
    edm_dir = PRETRAINED_MODEL_DIR / "e3_diffusion_for_molecules"
    ckpt = edm_dir / "outputs" / "edm_qm9" / "generative_model_ema.npy"
    args = edm_dir / "outputs" / "edm_qm9" / "args.pickle"

    from pretrained_models.export_edm import export_edm

    args, model = export_edm(ckpt, args, device=device)
    return args, model


def replace_mol_topology_by_fragment(mol: Chem.Mol, fragment: Chem.Mol, atom_indices: list[int]) -> Chem.Mol:
    """
    Replace topology of a subgraph in a molecule with a given fragment, preserving conformer coordinates.

    Args:
        mol: Target RDKit molecule object to modify
        fragment: Fragment RDKit molecule to use as replacement
        atom_indices: List of atom indices in mol defining the subgraph to replace

    Returns:
        RDKit molecule with replaced topology and preserved coordinates

    Raises:
        ValueError: If atom indices are invalid or fragment size doesn't match
    """
    if len(atom_indices) != fragment.GetNumAtoms():
        raise ValueError(
            f"Number of atoms in fragment ({fragment.GetNumAtoms()}) must match "
            f"number of target atoms ({len(atom_indices)})"
        )

    # Validate atom indices
    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range (0-{num_atoms - 1})")

    # Create editable molecule
    rw_mol = Chem.RWMol(mol)

    # Store original conformer if exists
    has_conf = mol.GetNumConformers() > 0
    if has_conf:
        orig_conf = mol.GetConformer()

    # Replace atoms while preserving coordinates
    for i, target_idx in enumerate(atom_indices):
        frag_atom = fragment.GetAtomWithIdx(i)
        target_atom = rw_mol.GetAtomWithIdx(target_idx)

        # Update atom properties
        target_atom.SetAtomicNum(frag_atom.GetAtomicNum())
        target_atom.SetFormalCharge(frag_atom.GetFormalCharge())
        target_atom.SetNumExplicitHs(frag_atom.GetNumExplicitHs())
        target_atom.SetIsAromatic(frag_atom.GetIsAromatic())
        target_atom.SetChiralTag(frag_atom.GetChiralTag())

    # Remove existing bonds between target atoms
    for i, idx1 in enumerate(atom_indices):
        for idx2 in atom_indices[i + 1 :]:
            bond = rw_mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is not None:
                rw_mol.RemoveBond(idx1, idx2)

    # Add new bonds from fragment
    for bond in fragment.GetBonds():
        begin_idx = atom_indices[bond.GetBeginAtomIdx()]
        end_idx = atom_indices[bond.GetEndAtomIdx()]
        rw_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

    # Convert back to molecule
    new_mol = rw_mol.GetMol()

    new_mol.RemoveAllConformers()
    # Copy conformer if existed
    if has_conf:
        new_conf = Chem.Conformer(new_mol.GetNumAtoms())
        for i in range(new_mol.GetNumAtoms()):
            pos = orig_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        new_mol.AddConformer(new_conf)

    # try:
    #     Chem.Kekulize(new_mol)
    #     Chem.SanitizeMol(new_mol)
    # except Exception as e:
    #     print(e)
    #     return None
    return new_mol
