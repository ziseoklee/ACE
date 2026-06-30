import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist


def get_max_valence(atom):
    """
    Returns the maximum allowed valence for an atom based on its type.
    """
    atomic_num = atom.GetAtomicNum()

    # Common max valences (ignoring charge states for simplicity)
    if atomic_num == 1:
        return 1  # H
    if atomic_num == 6:
        return 4  # C
    if atomic_num == 7:
        return 3  # N (neutral)
    if atomic_num == 8:
        return 2  # O
    if atomic_num == 9:
        return 1  # F
    if atomic_num == 15:
        return 5  # P
    if atomic_num == 16:
        return 6  # S
    if atomic_num == 17:
        return 1  # Cl
    if atomic_num == 35:
        return 1  # Br
    if atomic_num == 53:
        return 1  # I

    # Default fallback from Periodic Table
    pt = Chem.GetPeriodicTable()
    return pt.GetDefaultValence(atomic_num)


def get_covalent_threshold(atom_a, atom_b):
    """
    Returns the sum of covalent radii plus a tolerance buffer.
    """
    pt = Chem.GetPeriodicTable()
    ra = pt.GetRcovalent(atom_a.GetAtomicNum())
    rb = pt.GetRcovalent(atom_b.GetAtomicNum())

    # Tolerance for connectivity detection
    tolerance = 0.45
    return ra + rb + tolerance


def estimate_bond_order(atom_a, atom_b, dist):
    """
    Estimates bond order based on distance relative to covalent radii sum.
    Heuristics:
      - Dist < 0.82 * sum_radii -> Triple
      - Dist < 0.92 * sum_radii -> Double
      - Otherwise -> Single
    """
    pt = Chem.GetPeriodicTable()
    ra = pt.GetRcovalent(atom_a.GetAtomicNum())
    rb = pt.GetRcovalent(atom_b.GetAtomicNum())
    sum_radii = ra + rb

    # Heuristics for bond order based on compression
    if dist < sum_radii * 0.82:
        return Chem.BondType.TRIPLE
    elif dist < sum_radii * 0.92:
        return Chem.BondType.DOUBLE
    else:
        return Chem.BondType.SINGLE


def get_bond_order_val(bond_type):
    """Helper to get integer value of bond order"""
    if bond_type == Chem.BondType.SINGLE:
        return 1.0
    if bond_type == Chem.BondType.DOUBLE:
        return 2.0
    if bond_type == Chem.BondType.TRIPLE:
        return 3.0
    if bond_type == Chem.BondType.AROMATIC:
        return 1.5
    return 1.0


def calculate_initial_valences(mol):
    """
    Calculates the initial valences for all atoms in the molecule
    based on the bonds currently present (e.g., from scaffold).
    """
    valences = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
    for bond in mol.GetBonds():
        idx0 = bond.GetBeginAtomIdx()
        idx1 = bond.GetEndAtomIdx()
        order = get_bond_order_val(bond.GetBondType())
        valences[idx0] += order
        valences[idx1] += order
    return valences


def get_current_valence(atom):
    """
    Manually calculates the current sum of bond orders for an atom.
    This avoids RDKit's 'calcExplicitValence' error during RWMol editing.
    """
    val = 0.0
    for bond in atom.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            val += 1.0
        elif bt == Chem.BondType.DOUBLE:
            val += 2.0
        elif bt == Chem.BondType.TRIPLE:
            val += 3.0
        elif bt == Chem.BondType.AROMATIC:
            val += 1.5
        else:
            val += 1.0  # Default fallback
    return val


# --- Union-Find Helper for Connectivity ---
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.count = n  # Number of components

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.count -= 1
            return True
        return False


def reconstruct_molecule_with_scaffold(input_mol, scaffold_mol):
    """
    Reconstructs a molecule from an input RDKit molecule (treating it as a point cloud)
    while enforcing a scaffold topology, inferring new bonds from geometry,
    and guaranteeing full graph connectivity.

    Args:
        input_mol (rdkit.Chem.Mol): Input molecule with atoms and 3D coordinates.
                                    EXISTING BONDS ARE IGNORED.
        scaffold_mol (rdkit.Chem.Mol): The conditioning scaffold molecule (topology).
                                       Assumes the FIRST N atoms in input_mol correspond
                                       exactly (by index) to the scaffold_mol.
    """

    # 0. Extract Data from Input Mol
    atom_types = [atom.GetAtomicNum() for atom in input_mol.GetAtoms()]

    try:
        coordinates = input_mol.GetConformer().GetPositions()
    except ValueError:
        raise ValueError("Input molecule must have a 3D conformer.")

    # 1. Setup the Editable Molecule (RWMol) - Fresh Instance
    mol = Chem.RWMol()
    for z in atom_types:
        mol.AddAtom(Chem.Atom(int(z)))

    conf = Chem.Conformer(len(atom_types))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
    mol.AddConformer(conf)

    num_atoms = mol.GetNumAtoms()

    # 2. Derive Scaffold Indices
    num_scaffold_atoms = scaffold_mol.GetNumAtoms()

    if num_atoms < num_scaffold_atoms:
        raise ValueError(f"Input molecule has fewer atoms ({num_atoms}) than the scaffold size ({num_scaffold_atoms}).")

    global_scaffold_indices = np.arange(num_scaffold_atoms)
    existing_bonds = set()

    # 3. PHASE 1: Enforce Hard Constraints (Scaffold Topology)
    scaffold_bonds = scaffold_mol.GetBonds()

    for bond in scaffold_bonds:
        local_i = bond.GetBeginAtomIdx()
        local_j = bond.GetEndAtomIdx()
        global_i = int(global_scaffold_indices[local_i])
        global_j = int(global_scaffold_indices[local_j])

        mol.AddBond(global_i, global_j, bond.GetBondType())
        pair = tuple(sorted((global_i, global_j)))
        existing_bonds.add(pair)

    # 4. PREPARE FOR PHASE 2
    max_valences = np.array([get_max_valence(mol.GetAtomWithIdx(i)) for i in range(num_atoms)])
    current_valences = calculate_initial_valences(mol)
    dist_matrix = cdist(coordinates, coordinates)

    # Identify potential geometric bonds
    potential_bonds = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair = (i, j)
            if pair in existing_bonds:
                continue

            dist = dist_matrix[i, j]
            atom_i = mol.GetAtomWithIdx(i)
            atom_j = mol.GetAtomWithIdx(j)

            threshold = get_covalent_threshold(atom_i, atom_j)

            if dist < threshold:
                potential_bonds.append((dist, i, j))

    potential_bonds.sort(key=lambda x: x[0])

    # 5. PHASE 2: Geometry-based Connection with Dynamic Valence Tracking
    for dist, i, j in potential_bonds:
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)

        proposed_type = estimate_bond_order(atom_i, atom_j, dist)
        proposed_val = get_bond_order_val(proposed_type)

        curr_val_i = current_valences[i]
        curr_val_j = current_valences[j]

        rem_i = max_valences[i] - curr_val_i
        rem_j = max_valences[j] - curr_val_j

        allowed_val = min(proposed_val, rem_i, rem_j)

        if allowed_val >= 1.0:
            final_type = Chem.BondType.SINGLE
            actual_order_val = 1.0

            if allowed_val >= 2.9:
                final_type = Chem.BondType.TRIPLE
                actual_order_val = 3.0
            elif allowed_val >= 1.9:
                final_type = Chem.BondType.DOUBLE
                actual_order_val = 2.0

            mol.AddBond(i, j, final_type)
            current_valences[i] += actual_order_val
            current_valences[j] += actual_order_val
            existing_bonds.add((i, j))

    # 6. PHASE 3: Ensure Full Connectivity
    # Initialize UnionFind with existing bonds
    uf = UnionFind(num_atoms)
    for bond in mol.GetBonds():  # type: ignore
        uf.union(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    if uf.count > 1:
        # Collect all potential edges between disconnected components
        candidate_edges = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if uf.find(i) != uf.find(j):
                    candidate_edges.append((dist_matrix[i, j], i, j))

        # Sort candidates by distance (closest first)
        candidate_edges.sort(key=lambda x: x[0])

        # Pass 3.1: Try to connect using VALID VALENCE bonds only
        # We limit search to a reasonable "stretch" distance (e.g. 2.0A) to prefer
        # valid chemical bonds over long-distance valid artifacts.
        # If it's short and valid, it's a great anchor.

        valid_stretch_limit = 2.0  # Angstroms

        # Note: We iterate a copy or index because we might modify UF in loop
        for dist, i, j in candidate_edges:
            if uf.count == 1:
                break
            if dist > valid_stretch_limit:
                break  # Stop looking for "nice" bonds if too far

            if uf.find(i) != uf.find(j):
                # Valence Check
                if (current_valences[i] + 1.0 <= max_valences[i]) and (current_valences[j] + 1.0 <= max_valences[j]):
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    current_valences[i] += 1.0
                    current_valences[j] += 1.0
                    existing_bonds.add((i, j))
                    uf.union(i, j)

        # Pass 3.2: Force connect remaining components (anchors)
        # If still disconnected, we MUST connect closest anchors regardless of valence.
        if uf.count > 1:
            for dist, i, j in candidate_edges:
                if uf.count == 1:
                    break

                if uf.find(i) != uf.find(j):
                    # Force Add Single Bond
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    # Update valence just for tracking (though it exceeds max)
                    current_valences[i] += 1.0
                    current_valences[j] += 1.0
                    existing_bonds.add((i, j))
                    uf.union(i, j)

    # 7. Final Sanitization
    try:
        Chem.SanitizeMol(mol)
    except ValueError as e:
        print(f"Warning: Sanitization failed partially. {e}")

    return mol.GetMol()


def fix_valence_issues(mol, num_scaffold_atoms):
    """
    Iteratively fixes valence issues in a molecule by reducing/removing bonds.
    Prioritizes removing/reducing non-scaffold bonds first.
    Preserves scaffold connectivity (never removes a scaffold single bond).
    Preserves graph connectivity (never removes a bond that bridges two fragments).

    Args:
        mol (rdkit.Chem.RWMol or Mol): The molecule to fix.
        num_scaffold_atoms (int): The number of atoms in the scaffold.
                                  Assumes first N atoms match the scaffold.
    """
    # Ensure editable
    if not isinstance(mol, Chem.RWMol):
        rw_mol = Chem.RWMol(mol)
    else:
        rw_mol = mol

    # 2. Iterative Fix Loop
    # We loop until no changes are made in a full pass
    max_passes = 50
    for _ in range(max_passes):
        changed_this_pass = False

        # Iterate over all atoms by index
        # We re-fetch atom object inside loop in case of RWMol invalidation (rare for bonds, but safe)
        for atom_idx in range(rw_mol.GetNumAtoms()):
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            max_val = get_max_valence(atom)

            # Reduce valence until satisfied for this specific atom
            while True:
                curr_val = get_current_valence(atom)
                if curr_val <= max_val:
                    break

                # Atom is over-valent. Find a bond to reduce.
                bonds = atom.GetBonds()
                candidates = []

                for bond in bonds:
                    b_idx0 = bond.GetBeginAtomIdx()
                    b_idx1 = bond.GetEndAtomIdx()

                    b_type = bond.GetBondType()

                    # Rule: Identify if bond is within scaffold indices
                    is_scaffold = (b_idx0 < num_scaffold_atoms) and (b_idx1 < num_scaffold_atoms)

                    # Rule: Cannot remove scaffold single bond
                    if is_scaffold and b_type == Chem.BondType.SINGLE:
                        continue

                    # Rule: Priority 1 (Non-scaffold=0), Priority 2 (Scaffold=1)
                    priority = 1 if is_scaffold else 0

                    order_val = get_bond_order_val(b_type)

                    candidates.append(
                        {
                            "priority": priority,
                            "order_val": order_val,
                            "atom0": b_idx0,
                            "atom1": b_idx1,
                            "type": b_type,
                        }
                    )

                if not candidates:
                    # No reducible bonds found (e.g., all neighbors are scaffold single bonds)
                    # Cannot fix this atom. Break inner loop for this atom.
                    break

                # Sort: Priority ASC (Non-scaffold first), then Order DESC (Reduce Triple before Double)
                candidates.sort(key=lambda x: (x["priority"], -x["order_val"]))

                modification_done = False

                # Try candidates in order. If one fails (e.g. causes fragmentation), try next.
                for target in candidates:
                    u, v = target["atom0"], target["atom1"]
                    old_type = target["type"]

                    # Determine new type
                    new_type = None
                    if old_type == Chem.BondType.TRIPLE:
                        new_type = Chem.BondType.DOUBLE
                    elif old_type == Chem.BondType.DOUBLE:
                        new_type = Chem.BondType.SINGLE
                    elif old_type == Chem.BondType.AROMATIC:
                        new_type = Chem.BondType.SINGLE
                    elif old_type == Chem.BondType.SINGLE:
                        new_type = None

                    if new_type is None:
                        # CHECK FOR FRAGMENTATION BEFORE REMOVAL
                        # 1. Temporarily Remove
                        rw_mol.RemoveBond(u, v)

                        # 2. BFS to check connectivity between u and v
                        # If u and v are still connected via other paths, it's safe.
                        # If not, it's a bridge, so we restore and skip.
                        queue = [u]
                        visited = {u}
                        is_connected = False
                        while queue:
                            curr = queue.pop(0)
                            if curr == v:
                                is_connected = True
                                break

                            c_atom = rw_mol.GetAtomWithIdx(curr)
                            for nbr in c_atom.GetNeighbors():
                                n_idx = nbr.GetIdx()
                                if n_idx not in visited:
                                    visited.add(n_idx)
                                    queue.append(n_idx)

                        if not is_connected:
                            # It was a bridge! Restore and Try Next Candidate
                            rw_mol.AddBond(u, v, old_type)
                            continue
                        else:
                            # Safe to remove
                            modification_done = True
                            break
                    else:
                        # Just reducing order, safe regarding connectivity
                        rw_mol.RemoveBond(u, v)
                        rw_mol.AddBond(u, v, new_type)
                        modification_done = True
                        break

                if modification_done:
                    changed_this_pass = True
                    # Re-fetch atom to check valence again
                    atom = rw_mol.GetAtomWithIdx(atom_idx)
                else:
                    # All candidates exhausted (e.g. all were bridges or fixed constraints)
                    break

        if not changed_this_pass:
            break

    # Final Sanitization attempt
    try:
        Chem.SanitizeMol(rw_mol)
    except Exception:
        pass

    return rw_mol.GetMol()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Test Case: Scaffold + Disconnected Fragments
    # We simulate a case where geometry places atoms far apart or valence blocks connection

    scaffold = Chem.MolFromSmiles("CC")

    input_mol = Chem.RWMol()
    # Atoms: 2 (Scaffold) + 2 (Fragment A) + 2 (Fragment B)
    atom_types = [6, 6, 8, 8, 7, 7]
    for z in atom_types:
        input_mol.AddAtom(Chem.Atom(int(z)))

    # Coordinates:
    # 0,1: Scaffold (Near origin)
    # 2,3: Oxygen fragment (Far away, dist ~5.0)
    # 4,5: Nitrogen fragment (Even further)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],  # Scaffold
            [5.0, 5.0, 0.0],
            [5.0, 6.2, 0.0],  # Fragment 1 (O-O)
            [-5.0, -5.0, 0.0],
            [-5.0, -6.2, 0.0],  # Fragment 2 (N-N)
        ]
    )

    conf = Chem.Conformer(len(atom_types))
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
    input_mol.AddConformer(conf)

    print("Reconstructing with Guaranteed Connectivity...")
    result_mol = reconstruct_molecule_with_scaffold(input_mol, scaffold)

    print(f"Result SMILES: {Chem.MolToSmiles(result_mol)}")

    frags = Chem.GetMolFrags(result_mol)
    print(f"Number of fragments: {len(frags)} (Expected: 1)")

    # Test Valence Fixer on the result
    print("\nRunning Valence Fixer...")
    fixed_mol = fix_valence_issues(result_mol, scaffold.GetNumAtoms())
    print(f"Fixed SMILES: {Chem.MolToSmiles(fixed_mol)}")
