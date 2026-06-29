from rdkit import Chem


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
