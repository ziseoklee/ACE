import torch
import matplotlib.pyplot as plt
import sys
import contextlib, importlib, os, sys
import pandas as pd
from pathlib import Path
import subprocess
import lmdb
import pickle
from typing import Union, List, Dict, Any, Optional, Iterator
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

@contextlib.contextmanager
def sandbox_import_dir(dir_path: str):
    """
    Temporarily prefer `dir_path` for imports and avoid reusing
    same-named modules from sys.modules. Restores everything on exit.
    """
    dir_path = os.fspath(dir_path)
    # Detect module base names present in the directory (to guard)
    guard = {Path(p).stem for p in os.listdir(dir_path)}
    # print(guard)
    # Stash any conflicting modules already loaded
    # print(f'utils in sys.modules: {sys.modules["utils"]}')
    stash = {name: sys.modules.pop(name)
             for name in list(sys.modules)
             if name in guard or any(name.split(".")[0] == g for g in guard)}
    # print(f'stash: {stash}')
    sys.path.insert(0, dir_path)
    cwd = os.getcwd()
    os.chdir(dir_path)
    sys.path = list(filter(lambda x: x != cwd, sys.path))
    importlib.invalidate_caches()
    before = set(sys.modules)

    try:
        yield
    finally:
        # Remove anything imported from this dir during the sandbox
        for name in list(sys.modules):
            if name not in before:
                sys.modules.pop(name, None)
        # assert 'utils' not in sys.modules
        # Restore stashed modules and sys.path
        
        os.chdir(cwd)
        sys.path.remove(dir_path)

        sys.path.insert(0, cwd)
        sys.modules.update(stash)
        # assert 'utils' in sys.modules
        importlib.invalidate_caches()


### padding tensor ###
def pad_tensor(x, pad_size, dim=0):
    padding_shape = list(x.shape)
    padding_shape[dim] = pad_size - x.shape[dim]
    return torch.cat([x, torch.zeros(padding_shape, device=x.device)], dim=dim)

def pad_tensor_list(x_list, pad_size, dim=0):
    return [pad_tensor(x, pad_size, dim) for x in x_list]

def pad_tensor_mask(x, mask, dim=0):
    padding_shape = list(x.shape)
    padding_shape[dim] = len(mask) - x.shape[dim]
    tmp = torch.cat([x, torch.zeros(padding_shape, device=x.device)], dim=dim)

    out_shape = list(x.shape)
    out_shape[dim] = len(mask)
    out = torch.zeros(out_shape, device=x.device)
    return out

### plot any number of samples ###
def plot_samples(samples_dict):
    for key, samples in samples_dict.items():
        plt.scatter(samples[:, 0], samples[:, 1], label=key)
    plt.legend()
    plt.show()



def get_project_root():
    return Path(f'{__file__}').parent.parent.resolve()

REPOS = {
    "DiffSBDD": "https://github.com/arneschneuing/DiffSBDD.git",
    "GeoDiff": "https://github.com/MinkaiXu/GeoDiff.git",
    "e3_diffusion_for_molecules": "https://github.com/ehoogeboom/e3_diffusion_for_molecules.git",
}


def ensure_repo(name: str, url: str) -> Path:
    """Clone repo into baseline/ if it does not already exist."""
    root = Path(f'{__file__}').parent.parent /'src' / 'pretrained_models'
    path = root / name
    if not path.exists():
        subprocess.run(["git", "clone", url, str(path)], check=True)
    return path


def load_diffsbdd(device='cpu') -> str:
    """Instantiate the DiffSBDD model and load checkpoint weights."""
    path = ensure_repo("DiffSBDD", REPOS["DiffSBDD"])
    ckpt = path / "checkpoints" / "crossdocked_fullatom_cond.ckpt"
    with sandbox_import_dir(path):
        from export import export_diffsbdd
        model = export_diffsbdd(ckpt, device=device)
    model.to(device)
    return model


def load_geodiff(device='cpu') -> str:
    """Instantiate the GeoDiff model and load checkpoint weights."""
    path = ensure_repo("GeoDiff", REPOS["GeoDiff"])
    ckpt = path / "log" / "model" / "checkpoints" / "qm9_default.pt"
    with sandbox_import_dir(path):
        from export import export_geodiff
        config, model = export_geodiff(ckpt, device=device)
    model.to(device)
    return config, model

def load_e3_diffusion(device='cpu') -> str:
    """Instantiate the EDM model and load checkpoint weights."""
    path = ensure_repo("e3_diffusion_for_molecules", REPOS["e3_diffusion_for_molecules"])
    ckpt = path / "outputs" / "edm_qm9" / "generative_model_ema.npy"
    args = path / "outputs" / "edm_qm9" / "args.pickle"
    with sandbox_import_dir(path):
        from export import export_edm
        model = export_edm(ckpt, args, device=device)
    return model


# DataFrame Utilities
def df_to_latex(df, save_path):
    latex = df.to_latex(
        multicolumn=True,     # combine top-level column headers
        multirow=True,        # combine repeated row index labels
        index=True,           # include row MultiIndex
        bold_rows=False,      # set True if you want bold index labels
        na_rep="",
        escape=True,          # set False if your labels include LaTeX
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex)

def df_to_csv(df, save_path):
    df.to_csv(save_path, index=True)    

def df_to_excel(df, save_path, sheet_name="Sheet1"):
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=True,          # include row MultiIndex as left columns
            merge_cells=True,    # merge repeated headers (default True)
            na_rep=""
        )


# LMDB Utilities
class LMDBReader:
    """
    A utility class for reading LMDB databases with various convenience methods.
    """
    
    def __init__(self, lmdb_path: Union[str, Path], map_size: int = 10 * (1024**3), readonly: bool = True):
        """
        Initialize LMDB reader.
        
        Args:
            lmdb_path: Path to the LMDB database
            map_size: Maximum size of the memory map (default: 10GB)
            readonly: Whether to open in read-only mode
        """
        self.lmdb_path = str(lmdb_path)
        self.map_size = map_size
        self.readonly = readonly
        self.db = None
        self.keys = None
        
    def _connect(self):
        """Establish database connection if not already connected."""
        if self.db is None:
            self.db = lmdb.open(
                self.lmdb_path,
                map_size=self.map_size,
                create=False,
                subdir=False,
                readonly=self.readonly,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.db.begin() as txn:
                self.keys = list(txn.cursor().iternext(values=False))
    
    def _close(self):
        """Close database connection."""
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None
    
    def __enter__(self):
        """Context manager entry."""
        self._connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._close()
    
    def __len__(self):
        """Get number of entries in the database."""
        self._connect()
        return len(self.keys)
    
    def get_keys(self) -> List[bytes]:
        """Get all keys in the database."""
        self._connect()
        return self.keys.copy()
    
    def get_key(self, idx: int) -> bytes:
        """Get key at specific index."""
        self._connect()
        if idx < 0 or idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range for database with {len(self.keys)} entries")
        return self.keys[idx]
    
    def get_value(self, key: Union[bytes, int]) -> bytes:
        """
        Get raw value for a key.
        
        Args:
            key: Either a bytes key or an integer index
            
        Returns:
            Raw bytes value
        """
        self._connect()
        if isinstance(key, int):
            key = self.get_key(key)
        
        with self.db.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key not found: {key}")
            return value
    
    def get_pickled_value(self, key: Union[bytes, int]) -> Any:
        """
        Get and unpickle value for a key.
        
        Args:
            key: Either a bytes key or an integer index
            
        Returns:
            Unpickled Python object
        """
        raw_value = self.get_value(key)
        return pickle.loads(raw_value)
    
    def get_item(self, idx: int) -> Any:
        """
        Get item at specific index (convenience method).
        
        Args:
            idx: Index of the item
            
        Returns:
            Unpickled Python object
        """
        return self.get_pickled_value(idx)
    
    def iter_items(self) -> Iterator[tuple]:
        """
        Iterate over all key-value pairs.
        
        Yields:
            Tuple of (key, unpickled_value)
        """
        self._connect()
        with self.db.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                yield key, pickle.loads(value)
    
    def iter_values(self) -> Iterator[Any]:
        """
        Iterate over all values (unpickled).
        
        Yields:
            Unpickled Python objects
        """
        for _, value in self.iter_items():
            yield value
    
    def get_batch(self, indices: List[int]) -> List[Any]:
        """
        Get multiple items by their indices.
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            List of unpickled objects
        """
        return [self.get_item(idx) for idx in indices]
    
    def get_by_key_pattern(self, pattern: str) -> List[tuple]:
        """
        Get items whose keys contain a specific pattern.
        
        Args:
            pattern: String pattern to search for in keys
            
        Returns:
            List of (key, value) tuples matching the pattern
        """
        results = []
        for key, value in self.iter_items():
            if pattern.encode() in key:
                results.append((key, value))
        return results

    def get_by_value_filter(self, filter_func) -> List[tuple]:
        """
        Get items whose keys contain a specific pattern.
        
        Args:
            pattern: String pattern to search for in keys
            
        Returns:
            List of (key, value) tuples matching the pattern
        """
        results = []
        for key, value in self.iter_items():
            if filter_func(value):
                results.append((key, value))
        return results

    def info(self) -> Dict[str, Any]:
        """
        Get database information.
        
        Returns:
            Dictionary with database statistics
        """
        self._connect()
        with self.db.begin() as txn:
            stat = txn.stat()
            return {
                'path': self.lmdb_path,
                'entries': stat['entries'],
                'page_size': stat['psize'],
                'tree_depth': stat['depth'],
                'branch_pages': stat['branch_pages'],
                'leaf_pages': stat['leaf_pages'],
                'overflow_pages': stat['overflow_pages'],
                'keys': len(self.keys) if self.keys else 0
            }


def read_lmdb(lmdb_path: Union[str, Path], **kwargs) -> LMDBReader:
    """
    Convenience function to create an LMDBReader instance.
    
    Args:
        lmdb_path: Path to the LMDB database
        **kwargs: Additional arguments passed to LMDBReader
        
    Returns:
        LMDBReader instance
    """
    return LMDBReader(lmdb_path, **kwargs)


def read_lmdb_item(lmdb_path: Union[str, Path], idx: int, **kwargs) -> Any:
    """
    Convenience function to read a single item from LMDB.
    
    Args:
        lmdb_path: Path to the LMDB database
        idx: Index of the item to read
        **kwargs: Additional arguments passed to LMDBReader
        
    Returns:
        Unpickled Python object
    """
    with LMDBReader(lmdb_path, **kwargs) as reader:
        return reader.get_item(idx)


def read_lmdb_all(lmdb_path: Union[str, Path], **kwargs) -> List[Any]:
    """
    Convenience function to read all items from LMDB.
    
    Args:
        lmdb_path: Path to the LMDB database
        **kwargs: Additional arguments passed to LMDBReader
        
    Returns:
        List of all unpickled objects
    """
    with LMDBReader(lmdb_path, **kwargs) as reader:
        return list(reader.iter_values())


# Molecular Subgraph Utilities
def get_molecular_subgraph(data: Data, subgraph_index: int, 
                          subgraph_indices: Optional[torch.Tensor] = None) -> Data:
    """
    Extract a subgraph molecule from a molecular graph data object.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        subgraph_index: Index of the subgraph to extract (if subgraph_indices is None)
        subgraph_indices: Optional tensor of atom indices to include in subgraph
        
    Returns:
        Data object containing the subgraph molecule
        
    Raises:
        ValueError: If subgraph_index is invalid or subgraph_indices is empty
    """
    if subgraph_indices is None:
        # Use subgraph_index to find connected components
        if not hasattr(data, 'subgraph_index'):
            # If no subgraph_index attribute, compute connected components
            nx_graph = to_networkx(data, to_undirected=True)
            connected_components = list(nx.connected_components(nx_graph))
            if subgraph_index >= len(connected_components):
                raise ValueError(f"Subgraph index {subgraph_index} out of range. "
                               f"Only {len(connected_components)} connected components found.")
            subgraph_indices = torch.tensor(list(connected_components[subgraph_index]), dtype=torch.long)
        else:
            # Use precomputed subgraph_index
            mask = data.subgraph_index == subgraph_index
            if not mask.any():
                raise ValueError(f"No atoms found for subgraph index {subgraph_index}")
            subgraph_indices = torch.where(mask)[0]
    
    if len(subgraph_indices) == 0:
        raise ValueError("Subgraph indices cannot be empty")
    
    # Create subgraph using PyTorch Geometric's subgraph function
    subgraph_data = subgraph(subgraph_indices, data.edge_index, data.edge_type, 
                            num_nodes=data.num_nodes, return_edge_mask=True)
    
    edge_index, edge_attr, edge_mask = subgraph_data
    
    # Create new Data object with subgraph
    subgraph_mol = Data()
    
    # Copy node attributes
    for key, value in data:
        if key == 'edge_index' or key == 'edge_type':
            continue
        if isinstance(value, torch.Tensor) and value.size(0) == data.num_nodes:
            subgraph_mol[key] = value[subgraph_indices]
        else:
            subgraph_mol[key] = value
    
    # Set edge attributes
    subgraph_mol.edge_index = edge_index
    if edge_attr is not None:
        subgraph_mol.edge_type = edge_attr
    
    # Add mapping information
    subgraph_mol.subgraph_mapping = subgraph_indices
    subgraph_mol.original_num_nodes = data.num_nodes
    
    return subgraph_mol


def get_connected_components(data: Data) -> List[torch.Tensor]:
    """
    Get all connected components of a molecular graph.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        
    Returns:
        List of tensors, each containing atom indices for a connected component
    """
    nx_graph = to_networkx(data, to_undirected=True)
    connected_components = list(nx.connected_components(nx_graph))
    return [torch.tensor(list(comp), dtype=torch.long) for comp in connected_components]


def get_subgraph_by_atom_indices(data: Data, atom_indices: Union[List[int], torch.Tensor]) -> Data:
    """
    Extract a subgraph molecule by specifying atom indices directly.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        atom_indices: List or tensor of atom indices to include in subgraph
        
    Returns:
        Data object containing the subgraph molecule
    """
    if isinstance(atom_indices, list):
        atom_indices = torch.tensor(atom_indices, dtype=torch.long)
    
    return get_molecular_subgraph(data, 0, atom_indices)


def get_subgraph_by_radius(data: Data, center_atom_idx: int, radius: float) -> Data:
    """
    Extract a subgraph molecule within a specified radius from a center atom.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        center_atom_idx: Index of the center atom
        radius: Radius in Angstroms
        
    Returns:
        Data object containing the subgraph molecule
    """
    if not hasattr(data, 'pos'):
        raise ValueError("Data object must have 'pos' attribute for radius-based subgraph extraction")
    
    center_pos = data.pos[center_atom_idx]
    distances = torch.norm(data.pos - center_pos, dim=1)
    atom_indices = torch.where(distances <= radius)[0]
    
    return get_molecular_subgraph(data, 0, atom_indices)


def get_subgraph_by_bfs(data: Data, start_atom_idx: int, max_depth: int = 3) -> Data:
    """
    Extract a subgraph molecule using breadth-first search from a starting atom.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        start_atom_idx: Index of the starting atom
        max_depth: Maximum depth for BFS traversal
        
    Returns:
        Data object containing the subgraph molecule
    """
    nx_graph = to_networkx(data, to_undirected=True)
    
    # Perform BFS
    visited = set()
    queue = [(start_atom_idx, 0)]  # (node, depth)
    visited.add(start_atom_idx)
    
    while queue:
        node, depth = queue.pop(0)
        if depth >= max_depth:
            continue
            
        for neighbor in nx_graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    atom_indices = torch.tensor(list(visited), dtype=torch.long)
    return get_molecular_subgraph(data, 0, atom_indices)


def get_largest_connected_component(data: Data) -> Data:
    """
    Extract the largest connected component from a molecular graph.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        
    Returns:
        Data object containing the largest connected component
    """
    components = get_connected_components(data)
    if not components:
        raise ValueError("No connected components found")
    
    # Find largest component
    largest_component = max(components, key=len)
    return get_molecular_subgraph(data, 0, largest_component)


def analyze_molecular_subgraphs(data: Data) -> Dict[str, Any]:
    """
    Analyze the subgraphs (connected components) in a molecular graph.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        
    Returns:
        Dictionary containing analysis results
    """
    components = get_connected_components(data)
    
    analysis = {
        'num_components': len(components),
        'component_sizes': [len(comp) for comp in components],
        'largest_component_size': max(len(comp) for comp in components) if components else 0,
        'smallest_component_size': min(len(comp) for comp in components) if components else 0,
        'component_indices': components
    }
    
    return analysis


def visualize_subgraph(data: Data, subgraph_index: int = 0, 
                      subgraph_indices: Optional[torch.Tensor] = None,
                      title: str = "Molecular Subgraph") -> None:
    """
    Visualize a molecular subgraph using matplotlib.
    
    Args:
        data: PyTorch Geometric Data object containing molecular graph
        subgraph_index: Index of the subgraph to visualize
        subgraph_indices: Optional tensor of atom indices to include in subgraph
        title: Title for the plot
    """
    subgraph_data = get_molecular_subgraph(data, subgraph_index, subgraph_indices)
    
    if not hasattr(subgraph_data, 'pos'):
        print("Cannot visualize: subgraph data has no position information")
        return
    
    pos = subgraph_data.pos.cpu().numpy()
    atom_types = subgraph_data.atom_type.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Plot atoms
    scatter = plt.scatter(pos[:, 0], pos[:, 1], c=atom_types, cmap='tab20', s=100, alpha=0.7)
    
    # Plot edges
    if hasattr(subgraph_data, 'edge_index'):
        edge_index = subgraph_data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            start_idx, end_idx = edge_index[:, i]
            plt.plot([pos[start_idx, 0], pos[end_idx, 0]], 
                    [pos[start_idx, 1], pos[end_idx, 1]], 'k-', alpha=0.5, linewidth=1)
    
    plt.colorbar(scatter, label='Atom Type')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f"{title} (Atoms: {len(atom_types)})")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


# Convenience functions for common subgraph operations
def extract_ligand_subgraph(data: Data) -> Data:
    """
    Extract ligand subgraph from protein-ligand complex data.
    Assumes ligand atoms have specific attributes or are the smaller component.
    
    Args:
        data: PyTorch Geometric Data object containing protein-ligand complex
        
    Returns:
        Data object containing the ligand subgraph
    """
    if hasattr(data, 'ligand_context_pos'):
        # If ligand positions are explicitly provided
        ligand_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        # This is a simplified approach - you might need to adjust based on your data structure
        return get_molecular_subgraph(data, 0, ligand_mask)
    else:
        # Fall back to largest connected component (assuming ligand is smaller)
        return get_largest_connected_component(data)


def extract_protein_subgraph(data: Data) -> Data:
    """
    Extract protein subgraph from protein-ligand complex data.
    
    Args:
        data: PyTorch Geometric Data object containing protein-ligand complex
        
    Returns:
        Data object containing the protein subgraph
    """
    if hasattr(data, 'protein_pos'):
        # If protein positions are explicitly provided
        protein_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        # This is a simplified approach - you might need to adjust based on your data structure
        return get_molecular_subgraph(data, 0, protein_mask)
    else:
        # Fall back to largest connected component (assuming protein is larger)
        components = get_connected_components(data)
        if len(components) < 2:
            return data  # Only one component
        # Return the largest component (assuming it's the protein)
        largest_component = max(components, key=len)
        return get_molecular_subgraph(data, 0, largest_component)


# RDKit Molecular Subgraph Utilities
def get_rdkit_subgraph_by_atom_indices(mol: Chem.Mol, atom_indices: Union[List[int], np.ndarray]) -> Chem.Mol:
    """
    Extract a subgraph molecule from RDKit mol by specifying atom indices.
    
    Args:
        mol: RDKit molecule object
        atom_indices: List or array of atom indices to include in subgraph
        
    Returns:
        RDKit molecule object containing the subgraph
        
    Raises:
        ValueError: If atom_indices is empty or contains invalid indices
    """
    if isinstance(atom_indices, np.ndarray):
        atom_indices = atom_indices.tolist()
    
    if not atom_indices:
        raise ValueError("Atom indices cannot be empty")
    
    # Validate atom indices
    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range (0-{num_atoms-1})")
    
    # Create subgraph using RDKit's GetSubstructMatch approach
    # First, create a molecule with only the specified atoms
    rw_mol = Chem.RWMol()
    
    # Add atoms
    atom_map = {}
    for i, atom_idx in enumerate(atom_indices):
        atom = mol.GetAtomWithIdx(atom_idx)
        new_atom_idx = rw_mol.AddAtom(atom)
        atom_map[atom_idx] = new_atom_idx
    
    # Add bonds between atoms in the subgraph
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in atom_indices and neighbor_idx > atom_idx:  # Avoid duplicates
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                if bond is not None:
                    rw_mol.AddBond(atom_map[atom_idx], atom_map[neighbor_idx], bond.GetBondType())
    
    # Convert back to molecule
    subgraph_mol = rw_mol.GetMol()
    
    # Copy conformer if it exists
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        new_conf = Chem.Conformer(len(atom_indices))
        for i, atom_idx in enumerate(atom_indices):
            pos = conf.GetAtomPosition(atom_idx)
            new_conf.SetAtomPosition(i, pos)
        subgraph_mol.AddConformer(new_conf)
    
    return subgraph_mol

def replace_mol_topology_by_fragment(mol: Chem.Mol, fragment: Chem.Mol, atom_indices: List[int]) -> Chem.Mol:
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
        raise ValueError(f"Number of atoms in fragment ({fragment.GetNumAtoms()}) must match "
                       f"number of target atoms ({len(atom_indices)})")
        
    # Validate atom indices
    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range (0-{num_atoms-1})")
            
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
        for idx2 in atom_indices[i+1:]:
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


def get_rdkit_subgraph_by_radius(mol: Chem.Mol, center_atom_idx: int, radius: float) -> Chem.Mol:
    """
    Extract a subgraph molecule within a specified radius from a center atom.
    
    Args:
        mol: RDKit molecule object
        center_atom_idx: Index of the center atom
        radius: Radius in Angstroms
        
    Returns:
        RDKit molecule object containing the subgraph
        
    Raises:
        ValueError: If center_atom_idx is invalid or no conformer exists
    """
    if center_atom_idx < 0 or center_atom_idx >= mol.GetNumAtoms():
        raise ValueError(f"Center atom index {center_atom_idx} out of range")
    
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have conformer information for radius-based subgraph extraction")
    
    conf = mol.GetConformer()
    center_pos = conf.GetAtomPosition(center_atom_idx)
    
    # Find atoms within radius
    atom_indices = []
    for i in range(mol.GetNumAtoms()):
        atom_pos = conf.GetAtomPosition(i)
        distance = np.linalg.norm(np.array(atom_pos) - np.array(center_pos))
        if distance <= radius:
            atom_indices.append(i)
    
    if not atom_indices:
        raise ValueError(f"No atoms found within radius {radius} of center atom {center_atom_idx}")
    
    return get_rdkit_subgraph_by_atom_indices(mol, atom_indices)

def filter_valid_samples(mol_list):
    valid_samples = []
    for mol in mol_list:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)
            except Exception as e:
                continue
            if Chem.GetMolFrags(mol, asMols=True).__len__() != 1:
                continue
            valid_samples.append(mol)
    return valid_samples

def get_rdkit_subgraph_by_bfs(mol: Chem.Mol, start_atom_idx: int, max_depth: int = 3) -> Chem.Mol:
    """
    Extract a subgraph molecule using breadth-first search from a starting atom.
    
    Args:
        mol: RDKit molecule object
        start_atom_idx: Index of the starting atom
        max_depth: Maximum depth for BFS traversal
        
    Returns:
        RDKit molecule object containing the subgraph
        
    Raises:
        ValueError: If start_atom_idx is invalid
    """
    if start_atom_idx < 0 or start_atom_idx >= mol.GetNumAtoms():
        raise ValueError(f"Start atom index {start_atom_idx} out of range")
    
    # Perform BFS
    visited = set()
    queue = [(start_atom_idx, 0)]  # (atom_idx, depth)
    visited.add(start_atom_idx)
    
    while queue:
        atom_idx, depth = queue.pop(0)
        if depth >= max_depth:
            continue
            
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append((neighbor_idx, depth + 1))
    
    atom_indices = list(visited)
    return get_rdkit_subgraph_by_atom_indices(mol, atom_indices)


def get_rdkit_connected_components(mol: Chem.Mol) -> List[Chem.Mol]:
    """
    Get all connected components of an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of RDKit molecule objects, each representing a connected component
    """
    # Get molecule fragments
    frags = Chem.GetMolFrags(mol, asMols=True)
    return list(frags)


def get_rdkit_largest_connected_component(mol: Chem.Mol) -> Chem.Mol:
    """
    Extract the largest connected component from an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        RDKit molecule object containing the largest connected component
        
    Raises:
        ValueError: If no connected components found
    """
    components = get_rdkit_connected_components(mol)
    if not components:
        raise ValueError("No connected components found")
    
    # Find largest component
    largest_component = max(components, key=lambda x: x.GetNumAtoms())
    return largest_component

def get_rdkit_connected_component_containing(mol: Chem.Mol, atom_indices: List[int]) -> Chem.Mol:
    """
    Extract the connected component containing the specified atom indices.
    
    Args:
        mol: RDKit molecule object
        atom_indices: List of atom indices that must be included in the component
        
    Returns:
        RDKit molecule object containing the connected component with specified atoms
        
    Raises:
        ValueError: If atom indices are invalid or no connected component found
    """
    # Validate indices
    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range [0, {num_atoms})")
    
    # Get all connected components
    frags = Chem.GetMolFrags(mol, asMols=False)
    
    # Find component containing all specified atoms
    for frag in frags:
        if all(idx in frag for idx in atom_indices):
            return get_rdkit_subgraph_by_atom_indices(mol, list(frag))
            
    raise ValueError("No connected component contains all specified atom indices")


def get_rdkit_subgraph_by_smarts(mol: Chem.Mol, smarts_pattern: str) -> List[Chem.Mol]:
    """
    Extract subgraph molecules matching a SMARTS pattern.
    
    Args:
        mol: RDKit molecule object
        smarts_pattern: SMARTS pattern to match
        
    Returns:
        List of RDKit molecule objects matching the pattern
        
    Raises:
        ValueError: If SMARTS pattern is invalid
    """
    try:
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts_pattern}")
        
        matches = mol.GetSubstructMatches(pattern)
        subgraphs = []
        
        for match in matches:
            subgraph_mol = get_rdkit_subgraph_by_atom_indices(mol, list(match))
            subgraphs.append(subgraph_mol)
        
        return subgraphs
    except Exception as e:
        raise ValueError(f"Error processing SMARTS pattern '{smarts_pattern}': {e}")


def get_rdkit_subgraph_by_ring_system(mol: Chem.Mol, ring_idx: int = 0) -> Chem.Mol:
    """
    Extract a subgraph molecule containing a specific ring system.
    
    Args:
        mol: RDKit molecule object
        ring_idx: Index of the ring system to extract
        
    Returns:
        RDKit molecule object containing the ring system
        
    Raises:
        ValueError: If ring_idx is invalid
    """
    ring_info = mol.GetRingInfo()
    if ring_idx >= len(ring_info.AtomRings()):
        raise ValueError(f"Ring index {ring_idx} out of range. Molecule has {len(ring_info.AtomRings())} rings")
    
    ring_atoms = ring_info.AtomRings()[ring_idx]
    return get_rdkit_subgraph_by_atom_indices(mol, list(ring_atoms))


def analyze_rdkit_subgraphs(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Analyze the subgraphs (connected components) in an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary containing analysis results
    """
    components = get_rdkit_connected_components(mol)
    
    analysis = {
        'num_components': len(components),
        'component_sizes': [comp.GetNumAtoms() for comp in components],
        'largest_component_size': max(comp.GetNumAtoms() for comp in components) if components else 0,
        'smallest_component_size': min(comp.GetNumAtoms() for comp in components) if components else 0,
        'component_smiles': [Chem.MolToSmiles(comp) for comp in components],
        'total_atoms': mol.GetNumAtoms(),
        'total_bonds': mol.GetNumBonds()
    }
    
    return analysis


def rdkit_mol_to_networkx(mol: Chem.Mol) -> nx.Graph:
    """
    Convert RDKit molecule to NetworkX graph.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    
    # Add nodes (atoms)
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), 
                  symbol=atom.GetSymbol(),
                  atomic_num=atom.GetAtomicNum(),
                  formal_charge=atom.GetFormalCharge(),
                  hybridization=atom.GetHybridization())
    
    # Add edges (bonds)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                  bond_type=bond.GetBondType(),
                  is_aromatic=bond.GetIsAromatic())
    
    return G


def get_rdkit_subgraph_by_networkx_analysis(mol: Chem.Mol, analysis_func, **kwargs) -> Chem.Mol:
    """
    Extract subgraph using NetworkX analysis functions.
    
    Args:
        mol: RDKit molecule object
        analysis_func: NetworkX analysis function (e.g., nx.ego_graph, nx.subgraph)
        **kwargs: Arguments for the analysis function
        
    Returns:
        RDKit molecule object containing the subgraph
    """
    G = rdkit_mol_to_networkx(mol)
    
    # Apply analysis function
    subgraph_G = analysis_func(G, **kwargs)
    
    # Get atom indices from subgraph
    atom_indices = list(subgraph_G.nodes())
    
    return get_rdkit_subgraph_by_atom_indices(mol, atom_indices)


def get_rdkit_subgraph_by_ego_graph(mol: Chem.Mol, center_atom_idx: int, radius: int = 1) -> Chem.Mol:
    """
    Extract ego graph (subgraph within radius) from RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        center_atom_idx: Index of the center atom
        radius: Radius for ego graph
        
    Returns:
        RDKit molecule object containing the ego graph
    """
    return get_rdkit_subgraph_by_networkx_analysis(
        mol, nx.ego_graph, n=center_atom_idx, radius=radius
    )


def get_rdkit_subgraph_by_centrality(mol: Chem.Mol, centrality_measure: str = 'betweenness', 
                                   top_k: int = 10) -> Chem.Mol:
    """
    Extract subgraph containing top-k most central atoms.
    
    Args:
        mol: RDKit molecule object
        centrality_measure: Centrality measure ('betweenness', 'closeness', 'eigenvector', 'degree')
        top_k: Number of top central atoms to include
        
    Returns:
        RDKit molecule object containing the subgraph
        
    Raises:
        ValueError: If centrality_measure is not supported
    """
    G = rdkit_mol_to_networkx(mol)
    
    # Calculate centrality
    if centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_measure == 'eigenvector':
        centrality = nx.eigenvector_centrality(G)
    elif centrality_measure == 'degree':
        centrality = nx.degree_centrality(G)
    else:
        raise ValueError(f"Unsupported centrality measure: {centrality_measure}")
    
    # Get top-k central atoms
    top_atoms = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
    atom_indices = [atom_idx for atom_idx, _ in top_atoms]
    
    return get_rdkit_subgraph_by_atom_indices(mol, atom_indices)


# RDKit Conformation Utilities
def replace_mol_conformation_with_numpy(mol: Chem.Mol, positions: np.ndarray, conf_id: int = 0) -> Chem.Mol:
    """
    Replace molecule conformation with numpy array positions.
    
    Args:
        mol: RDKit molecule object
        positions: Numpy array of shape (n_atoms, 3) containing xyz coordinates
        conf_id: Conformer ID to replace (default: 0)
        
    Returns:
        RDKit molecule object with updated conformation
        
    Raises:
        ValueError: If positions array has wrong shape or conformer doesn't exist
    """
    if positions.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(f"Positions array must have shape ({mol.GetNumAtoms()}, 3), got {positions.shape}")
    
    # Check if conformer exists
    if conf_id >= mol.GetNumConformers():
        raise ValueError(f"Conformer {conf_id} does not exist. Molecule has {mol.GetNumConformers()} conformers")
    
    # Get the conformer
    conf = mol.GetConformer(conf_id)
    
    # Update positions
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
    
    return mol


def add_mol_conformation_from_numpy(mol: Chem.Mol, positions: np.ndarray, conf_id: Optional[int] = None) -> Chem.Mol:
    """
    Add a new conformation to molecule from numpy array positions.
    
    Args:
        mol: RDKit molecule object
        positions: Numpy array of shape (n_atoms, 3) containing xyz coordinates
        conf_id: Optional conformer ID to use (if None, uses next available ID)
        
    Returns:
        RDKit molecule object with new conformation added
        
    Raises:
        ValueError: If positions array has wrong shape
    """
    if positions.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(f"Positions array must have shape ({mol.GetNumAtoms()}, 3), got {positions.shape}")
    
    # Create new conformer
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    
    # Set positions
    for i in range(mol.GetNumAtoms()):
        new_conf.SetAtomPosition(i, (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
    
    # Add conformer to molecule
    if conf_id is not None:
        new_conf.SetId(conf_id)
        mol.AddConformer(new_conf, assignId=False)
    else:
        mol.AddConformer(new_conf)
    
    return mol


def get_mol_conformation_as_numpy(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    """
    Get molecule conformation as numpy array.
    
    Args:
        mol: RDKit molecule object
        conf_id: Conformer ID to extract (default: 0)
        
    Returns:
        Numpy array of shape (n_atoms, 3) containing xyz coordinates
        
    Raises:
        ValueError: If conformer doesn't exist
    """
    if conf_id >= mol.GetNumConformers():
        raise ValueError(f"Conformer {conf_id} does not exist. Molecule has {mol.GetNumConformers()} conformers")
    
    conf = mol.GetConformer(conf_id)
    positions = np.zeros((mol.GetNumAtoms(), 3))
    
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions[i, 0] = pos.x
        positions[i, 1] = pos.y
        positions[i, 2] = pos.z
    
    return positions


def replace_all_mol_conformations_with_numpy(mol: Chem.Mol, positions_list: List[np.ndarray]) -> Chem.Mol:
    """
    Replace all molecule conformations with numpy array positions.
    
    Args:
        mol: RDKit molecule object
        positions_list: List of numpy arrays, each of shape (n_atoms, 3)
        
    Returns:
        RDKit molecule object with all conformations updated
        
    Raises:
        ValueError: If positions arrays have wrong shapes or count mismatch
    """
    if len(positions_list) != mol.GetNumConformers():
        raise ValueError(f"Number of position arrays ({len(positions_list)}) must match number of conformers ({mol.GetNumConformers()})")
    
    for conf_id, positions in enumerate(positions_list):
        mol = replace_mol_conformation_with_numpy(mol, positions, conf_id)
    
    return mol


def create_mol_with_numpy_conformation(smiles: str, positions: np.ndarray) -> Chem.Mol:
    """
    Create a new molecule from SMILES and add conformation from numpy array.
    
    Args:
        smiles: SMILES string
        positions: Numpy array of shape (n_atoms, 3) containing xyz coordinates
        
    Returns:
        RDKit molecule object with conformation
        
    Raises:
        ValueError: If SMILES is invalid or positions array has wrong shape
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    if positions.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(f"Positions array must have shape ({mol.GetNumAtoms()}, 3), got {positions.shape}")
    
    # Add conformer
    mol = add_mol_conformation_from_numpy(mol, positions)
    
    return mol


def update_mol_conformation_from_tensor(mol: Chem.Mol, positions: torch.Tensor, conf_id: int = 0) -> Chem.Mol:
    """
    Update molecule conformation from PyTorch tensor positions.
    
    Args:
        mol: RDKit molecule object
        positions: PyTorch tensor of shape (n_atoms, 3) containing xyz coordinates
        conf_id: Conformer ID to update (default: 0)
        
    Returns:
        RDKit molecule object with updated conformation
        
    Raises:
        ValueError: If positions tensor has wrong shape or conformer doesn't exist
    """
    # Convert tensor to numpy
    if isinstance(positions, torch.Tensor):
        positions_np = positions.detach().cpu().numpy()
    else:
        positions_np = positions
    
    return replace_mol_conformation_with_numpy(mol, positions_np, conf_id)


def batch_update_mol_conformations(mols: List[Chem.Mol], positions_batch: np.ndarray, conf_id: int = 0) -> List[Chem.Mol]:
    """
    Update conformations for a batch of molecules.
    
    Args:
        mols: List of RDKit molecule objects
        positions_batch: Numpy array of shape (batch_size, n_atoms, 3) containing xyz coordinates
        conf_id: Conformer ID to update (default: 0)
        
    Returns:
        List of RDKit molecule objects with updated conformations
        
    Raises:
        ValueError: If batch size doesn't match or positions have wrong shapes
    """
    if len(mols) != positions_batch.shape[0]:
        raise ValueError(f"Number of molecules ({len(mols)}) must match batch size ({positions_batch.shape[0]})")
    
    updated_mols = []
    for i, mol in enumerate(mols):
        positions = positions_batch[i]
        updated_mol = replace_mol_conformation_with_numpy(mol, positions, conf_id)
        updated_mols.append(updated_mol)
    
    return updated_mols


def copy_conformation_between_mols(source_mol: Chem.Mol, target_mol: Chem.Mol, 
                                 source_conf_id: int = 0, target_conf_id: int = 0,
                                 atom_mapping: Optional[List[int]] = None) -> Chem.Mol:
    """
    Copy conformation from one molecule to another.
    
    Args:
        source_mol: Source molecule to copy conformation from
        target_mol: Target molecule to copy conformation to
        source_conf_id: Source conformer ID (default: 0)
        target_conf_id: Target conformer ID (default: 0)
        atom_mapping: Optional mapping from target atom indices to source atom indices
        
    Returns:
        Target molecule with copied conformation
        
    Raises:
        ValueError: If conformers don't exist or atom mapping is invalid
    """
    if source_conf_id >= source_mol.GetNumConformers():
        raise ValueError(f"Source conformer {source_conf_id} does not exist")
    
    if target_conf_id >= target_mol.GetNumConformers():
        raise ValueError(f"Target conformer {target_conf_id} does not exist")
    
    source_conf = source_mol.GetConformer(source_conf_id)
    target_conf = target_mol.GetConformer(target_conf_id)
    
    if atom_mapping is None:
        # Direct copy (assumes same number of atoms and same order)
        if source_mol.GetNumAtoms() != target_mol.GetNumAtoms():
            raise ValueError("Molecules must have same number of atoms for direct copy")
        
        for i in range(target_mol.GetNumAtoms()):
            pos = source_conf.GetAtomPosition(i)
            target_conf.SetAtomPosition(i, pos)
    else:
        # Copy with atom mapping
        if len(atom_mapping) != target_mol.GetNumAtoms():
            raise ValueError(f"Atom mapping length ({len(atom_mapping)}) must match target molecule atoms ({target_mol.GetNumAtoms()})")
        
        for i, source_idx in enumerate(atom_mapping):
            if source_idx < 0 or source_idx >= source_mol.GetNumAtoms():
                raise ValueError(f"Invalid source atom index {source_idx} in mapping")
            pos = source_conf.GetAtomPosition(source_idx)
            target_conf.SetAtomPosition(i, pos)
    
    return target_mol


def align_mol_conformation_to_reference(mol: Chem.Mol, reference_mol: Chem.Mol,
                                      mol_conf_id: int = 0, ref_conf_id: int = 0,
                                      atom_mapping: Optional[List[int]] = None) -> Chem.Mol:
    """
    Align molecule conformation to a reference molecule using Kabsch algorithm.
    
    Args:
        mol: Molecule to align
        reference_mol: Reference molecule
        mol_conf_id: Conformer ID of molecule to align (default: 0)
        ref_conf_id: Conformer ID of reference molecule (default: 0)
        atom_mapping: Optional mapping from mol atom indices to reference atom indices
        
    Returns:
        Aligned molecule
        
    Raises:
        ValueError: If conformers don't exist or atom mapping is invalid
    """
    from scipy.spatial.transform import Rotation
    
    # Get positions
    mol_positions = get_mol_conformation_as_numpy(mol, mol_conf_id)
    ref_positions = get_mol_conformation_as_numpy(reference_mol, ref_conf_id)
    
    if atom_mapping is not None:
        if len(atom_mapping) != mol.GetNumAtoms():
            raise ValueError(f"Atom mapping length ({len(atom_mapping)}) must match molecule atoms ({mol.GetNumAtoms()})")
        mol_positions = mol_positions[atom_mapping]
    
    if mol_positions.shape != ref_positions.shape:
        raise ValueError(f"Position arrays must have same shape: {mol_positions.shape} vs {ref_positions.shape}")
    
    # Center both sets of points
    mol_centered = mol_positions - mol_positions.mean(axis=0)
    ref_centered = ref_positions - ref_positions.mean(axis=0)
    
    # Compute rotation matrix using Kabsch algorithm
    H = mol_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation
    mol_aligned = (mol_positions - mol_positions.mean(axis=0)) @ R.T + ref_positions.mean(axis=0)
    
    # Update molecule conformation
    return replace_mol_conformation_with_numpy(mol, mol_aligned, mol_conf_id)


def get_conformation_rmsd(mol1: Chem.Mol, mol2: Chem.Mol, 
                         conf1_id: int = 0, conf2_id: int = 0,
                         atom_mapping: Optional[List[int]] = None) -> float:
    """
    Calculate RMSD between two molecule conformations.
    
    Args:
        mol1: First molecule
        mol2: Second molecule
        conf1_id: Conformer ID of first molecule (default: 0)
        conf2_id: Conformer ID of second molecule (default: 0)
        atom_mapping: Optional mapping from mol1 atom indices to mol2 atom indices
        
    Returns:
        RMSD value
        
    Raises:
        ValueError: If conformers don't exist or atom mapping is invalid
    """
    # Get positions
    pos1 = get_mol_conformation_as_numpy(mol1, conf1_id)
    pos2 = get_mol_conformation_as_numpy(mol2, conf2_id)
    
    if atom_mapping is not None:
        if len(atom_mapping) != mol1.GetNumAtoms():
            raise ValueError(f"Atom mapping length ({len(atom_mapping)}) must match mol1 atoms ({mol1.GetNumAtoms()})")
        pos1 = pos1[atom_mapping]
    
    if pos1.shape != pos2.shape:
        raise ValueError(f"Position arrays must have same shape: {pos1.shape} vs {pos2.shape}")
    
    # Calculate RMSD
    diff = pos1 - pos2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return float(rmsd)



def _to_mol(mol_or_smiles: Union[str, Chem.Mol]) -> Chem.Mol:
    if isinstance(mol_or_smiles, Chem.Mol):
        mol = Chem.Mol(mol_or_smiles)
    else:
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError("Cannot parse SMILES.")
    Chem.SanitizeMol(mol)
    return mol

def get_submolecule_smiles_with_dummy_atoms(submolecule: Chem.Mol, complete_molecule: Chem.Mol) -> str:
    """
    Get SMILES of a submolecule with dummy atoms representing the closest external atoms.
    
    Instead of replacing subgraph atoms, this function preserves the submolecule and
    ATTACHES dummy atoms (*) to those submolecule atoms that are bonded to atoms outside
    the subgraph in the complete molecule. The bond order to each dummy matches the
    original bond to the external atom. Dummy atoms are labeled with atom-map numbers
    corresponding to the external atom indices in the complete molecule (1-based) to
    uniquely identify attachment points.
    
    Args:
        submolecule: RDKit molecule object representing the submolecule
        complete_molecule: RDKit molecule object representing the complete molecule
        
    Returns:
        SMILES string of the submolecule with attached dummy atoms at external connections
        
    Raises:
        ValueError: If submolecule is not a valid substructure of complete_molecule
    """
    if not complete_molecule.HasSubstructMatch(submolecule):
        raise ValueError("Submolecule is not a substructure of the complete molecule")
    
    match = complete_molecule.GetSubstructMatch(submolecule)
    if not match:
        raise ValueError("Could not find submolecule in complete molecule")
    
    # Map complete molecule atom index -> submolecule atom index
    complete_to_sub = {complete_idx: sub_idx for sub_idx, complete_idx in enumerate(match)}
    
    rw_submol = Chem.RWMol(submolecule)
    added_dummies = 0
    
    for sub_atom_idx in range(submolecule.GetNumAtoms()):
        complete_atom_idx = match[sub_atom_idx]
        complete_atom = complete_molecule.GetAtomWithIdx(complete_atom_idx)
        
        for neighbor in complete_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in complete_to_sub:
                # External neighbor: add a dummy atom and connect with original bond type
                bond = complete_molecule.GetBondBetweenAtoms(complete_atom_idx, neighbor_idx)
                bond_type = bond.GetBondType() if bond is not None else Chem.BondType.SINGLE
                dummy = Chem.Atom(0)
                # No atom-map numbers set to avoid numbered dummy atoms in SMILES
                dummy_idx = rw_submol.AddAtom(dummy)
                rw_submol.AddBond(sub_atom_idx, dummy_idx, bond_type)
                added_dummies += 1
    
    modified_submol = rw_submol.GetMol()
    
    try:
        return Chem.MolToSmiles(modified_submol, canonical=True)
    except Exception:
        try:
            Chem.SanitizeMol(modified_submol)
            return Chem.MolToSmiles(modified_submol, canonical=True)
        except Exception:
            return f"Submolecule with {added_dummies} attached dummy atoms"


def get_submolecule_smiles_with_anchored_dummy_atoms(submolecule: Chem.Mol, complete_molecule: Chem.Mol, 
                                                   dummy_atom_symbol: str = "*") -> str:
    """
    Get SMILES representation of a submolecule with anchored dummy atoms.
    
    This creates an explicit representation by adding one or more dummy atoms to each
    anchor atom in the subgraph for every bond to an external atom. The original bond
    order is preserved. Dummy atoms are labeled with atom-map numbers corresponding to
    the external atom indices in the complete molecule (1-based).
    
    Args:
        submolecule: RDKit molecule object representing the submolecule
        complete_molecule: RDKit molecule object representing the complete molecule
        dummy_atom_symbol: Symbol to use for dummy atoms (default: "*")
        
    Returns:
        SMILES string of the submolecule with anchored dummy atoms
        
    Raises:
        ValueError: If submolecule is not a valid substructure of complete_molecule
    """
    # First, verify that submolecule is a substructure of complete_molecule
    if not complete_molecule.HasSubstructMatch(submolecule):
        raise ValueError("Submolecule is not a substructure of the complete molecule")
    
    # Get the atom mapping from complete molecule to submolecule
    match = complete_molecule.GetSubstructMatch(submolecule)
    if not match:
        raise ValueError("Could not find submolecule in complete molecule")
    
    # Create a mapping from complete molecule atom indices to submolecule atom indices
    complete_to_sub = {complete_idx: sub_idx for sub_idx, complete_idx in enumerate(match)}
    
    # Create a copy of the submolecule to modify
    rw_submol = Chem.RWMol(submolecule)
    
    # Add dummy atoms connected to anchor atoms, preserving bond order and labeling
    dummy_count = 0
    for sub_atom_idx in range(submolecule.GetNumAtoms()):
        complete_atom_idx = match[sub_atom_idx]
        complete_atom = complete_molecule.GetAtomWithIdx(complete_atom_idx)
        for neighbor in complete_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in complete_to_sub:
                bond = complete_molecule.GetBondBetweenAtoms(complete_atom_idx, neighbor_idx)
                bond_type = bond.GetBondType() if bond is not None else Chem.BondType.SINGLE
                dummy = Chem.Atom(0)
                try:
                    dummy.SetAtomMapNum(int(neighbor_idx) + 1)
                except Exception:
                    pass
                new_idx = rw_submol.AddAtom(dummy)
                rw_submol.AddBond(sub_atom_idx, new_idx, bond_type)
                dummy_count += 1
    
    # Convert back to molecule
    modified_submol = rw_submol.GetMol()
    
    # Generate SMILES with dummy atoms
    try:
        smiles = Chem.MolToSmiles(modified_submol, canonical=True)
        return smiles
    except Exception as e:
        # If SMILES generation fails, try to sanitize first
        try:
            Chem.SanitizeMol(modified_submol)
            smiles = Chem.MolToSmiles(modified_submol, canonical=True)
            return smiles
        except Exception:
            # If still fails, return a basic representation
            return f"Submolecule with {dummy_count} anchored dummy atoms"


def get_submolecule_connection_info(submolecule: Chem.Mol, complete_molecule: Chem.Mol) -> Dict[str, Any]:
    """
    Get detailed information about how a submolecule connects to the complete molecule.
    
    Args:
        submolecule: RDKit molecule object representing the submolecule
        complete_molecule: RDKit molecule object representing the complete molecule
        
    Returns:
        Dictionary containing connection information including:
        - connection_points: List of atoms in submolecule that have external connections
        - external_connections: Number of external connections per connection point
        - bond_types: Types of bonds to external atoms
        - smiles_with_dummy: SMILES with dummy atoms at connection points
    """
    # First, verify that submolecule is a substructure of complete_molecule
    if not complete_molecule.HasSubstructMatch(submolecule):
        raise ValueError("Submolecule is not a substructure of the complete molecule")
    
    # Get the atom mapping from complete molecule to submolecule
    match = complete_molecule.GetSubstructMatch(submolecule)
    if not match:
        raise ValueError("Could not find submolecule in complete molecule")
    
    # Create a mapping from complete molecule atom indices to submolecule atom indices
    complete_to_sub = {complete_idx: sub_idx for sub_idx, complete_idx in enumerate(match)}
    
    connection_info = {
        'connection_points': [],
        'external_connections': {},
        'bond_types': {},
        'smiles_with_dummy': get_submolecule_smiles_with_dummy_atoms(submolecule, complete_molecule)
    }
    
    # Analyze each atom in the submolecule
    for sub_atom_idx in range(submolecule.GetNumAtoms()):
        complete_atom_idx = match[sub_atom_idx]
        complete_atom = complete_molecule.GetAtomWithIdx(complete_atom_idx)
        
        # Find external connections
        external_bonds = []
        for neighbor in complete_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in complete_to_sub:
                # This is an external connection
                bond = complete_molecule.GetBondBetweenAtoms(complete_atom_idx, neighbor_idx)
                external_bonds.append({
                    'external_atom_idx': neighbor_idx,
                    'bond_type': bond.GetBondType() if bond else None,
                    'external_atom_symbol': complete_molecule.GetAtomWithIdx(neighbor_idx).GetSymbol()
                })
        
        if external_bonds:
            connection_info['connection_points'].append(sub_atom_idx)
            connection_info['external_connections'][sub_atom_idx] = len(external_bonds)
            connection_info['bond_types'][sub_atom_idx] = [bond['bond_type'] for bond in external_bonds]
    
    return connection_info


def sample_random_connected_subgraph(mol: Chem.Mol, min_size: int = 2, max_size: Optional[int] = None, 
                                    start_atom_idx: Optional[int] = None, initial_indices: Optional[List[int]] = None, include_atom_types: Optional[List[str]] = None, random_seed: Optional[int] = None) -> Chem.Mol:
    """
    Sample a random connected subgraph from an RDKit molecule.
    
    This function uses a breadth-first search approach to grow a connected subgraph
    by randomly selecting neighbors at each step. The resulting subgraph is guaranteed
    to be connected.
    
    Args:
        mol: RDKit molecule object
        min_size: Minimum number of atoms in the subgraph (default: 2)
        max_size: Maximum number of atoms in the subgraph (default: None, no limit)
        start_atom_idx: Starting atom index (default: None, random selection)
        random_seed: Random seed for reproducibility (default: None)
        
    Returns:
        RDKit molecule object containing the random connected subgraph
        
    Raises:
        ValueError: If min_size is invalid or molecule has no atoms
    """
    if mol.GetNumAtoms() == 0:
        raise ValueError("Molecule has no atoms")
    
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    
    if max_size is None:
        max_size = mol.GetNumAtoms()
    else:
        max_size = min(max_size, mol.GetNumAtoms())
    
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot be greater than max_size ({max_size})")
    
    max_size = np.random.randint(min_size, max_size + 1)
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Select starting atom
    if start_atom_idx is None:
        if initial_indices is None:
            while True:
                start_atom_idx = np.random.randint(0, mol.GetNumAtoms())
                if include_atom_types is not None and mol.GetAtomWithIdx(start_atom_idx).GetSymbol() in include_atom_types:
                    break
                if include_atom_types is None:
                    break
        else:
            start_atom_idx = initial_indices[0]
    elif start_atom_idx < 0 or start_atom_idx >= mol.GetNumAtoms():
        raise ValueError(f"start_atom_idx {start_atom_idx} out of range [0, {mol.GetNumAtoms()-1}]")
    
    # Initialize subgraph with starting atom
    subgraph_atoms = set(initial_indices) if initial_indices is not None else {start_atom_idx}
    candidate_neighbors = set()
    
    # Add initial neighbors to candidate set
    start_atom = mol.GetAtomWithIdx(start_atom_idx)
    for neighbor in start_atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx not in subgraph_atoms:
            candidate_neighbors.add(neighbor_idx)
    
    # Grow subgraph randomly
    while len(subgraph_atoms) < min_size or (len(candidate_neighbors) > 0 and len(subgraph_atoms) < max_size):
        if not candidate_neighbors:
            break
        
        # Randomly select a candidate neighbor
        selected_neighbor = np.random.choice(list(candidate_neighbors)).tolist()
        if include_atom_types is not None and mol.GetAtomWithIdx(selected_neighbor).GetSymbol() not in include_atom_types:
            candidate_neighbors.remove(selected_neighbor)
            continue
        subgraph_atoms.add(selected_neighbor)
        candidate_neighbors.remove(selected_neighbor)
        
        # Add new neighbors to candidate set
        selected_atom = mol.GetAtomWithIdx(selected_neighbor)
        for neighbor in selected_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in subgraph_atoms:
                candidate_neighbors.add(neighbor_idx)
    
    # Convert to list of integers and create subgraph
    atom_indices = list(subgraph_atoms)
    return get_rdkit_subgraph_by_atom_indices(mol, atom_indices)

def sample_multiple_random_connected_subgraphs(mol: Chem.Mol, n_samples: int, 
                                             min_size: int = 2, max_size: Optional[int] = None, initial_indices: Optional[List[int]] = None, include_atom_types: Optional[List[str]] = None,
                                             random_seed: Optional[int] = None) -> List[Chem.Mol]:
    """
    Sample multiple random connected subgraphs from an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        n_samples: Number of subgraphs to sample
        min_size: Minimum number of atoms in each subgraph (default: 2)
        max_size: Maximum number of atoms in each subgraph (default: None, no limit)
        random_seed: Random seed for reproducibility (default: None)
        
    Returns:
        List of RDKit molecule objects containing the random connected subgraphs
        
    Raises:
        ValueError: If n_samples is invalid or molecule has no atoms
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    subgraphs = []
    for i in range(n_samples):
        # Use different seed for each sample if seed is provided
        sample_seed = random_seed + i if random_seed is not None else None
        subgraph = sample_random_connected_subgraph(
            mol, min_size=min_size, max_size=max_size, random_seed=sample_seed, initial_indices=initial_indices, include_atom_types=include_atom_types
        )
        subgraphs.append(subgraph)
    
    return subgraphs


def sample_weighted_random_connected_subgraph(mol: Chem.Mol, atom_weights: Optional[List[float]] = None,
                                            bond_weights: Optional[Dict[tuple, float]] = None,
                                            min_size: int = 2, max_size: Optional[int] = None,
                                            start_atom_idx: Optional[int] = None, 
                                            random_seed: Optional[int] = None) -> Chem.Mol:
    """
    Sample a random connected subgraph using atom and/or bond weights.
    
    This function allows for weighted sampling where atoms and bonds can have
    different probabilities of being selected during subgraph growth.
    
    Args:
        mol: RDKit molecule object
        atom_weights: Optional list of weights for each atom (higher = more likely to be selected)
        bond_weights: Optional dict mapping (atom1_idx, atom2_idx) tuples to bond weights
        min_size: Minimum number of atoms in the subgraph (default: 2)
        max_size: Maximum number of atoms in the subgraph (default: None, no limit)
        start_atom_idx: Starting atom index (default: None, weighted random selection)
        random_seed: Random seed for reproducibility (default: None)
        
    Returns:
        RDKit molecule object containing the weighted random connected subgraph
        
    Raises:
        ValueError: If weights are invalid or molecule has no atoms
    """
    if mol.GetNumAtoms() == 0:
        raise ValueError("Molecule has no atoms")
    
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    
    if max_size is None:
        max_size = mol.GetNumAtoms()
    else:
        max_size = min(max_size, mol.GetNumAtoms())
    
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot be greater than max_size ({max_size})")
    
    # Validate and normalize atom weights
    if atom_weights is not None:
        if len(atom_weights) != mol.GetNumAtoms():
            raise ValueError(f"atom_weights length ({len(atom_weights)}) must match number of atoms ({mol.GetNumAtoms()})")
        atom_weights = np.array(atom_weights)
        if np.any(atom_weights < 0):
            raise ValueError("atom_weights must be non-negative")
        # Normalize weights
        atom_weights = atom_weights / np.sum(atom_weights) if np.sum(atom_weights) > 0 else np.ones_like(atom_weights) / len(atom_weights)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Select starting atom
    if start_atom_idx is None:
        if atom_weights is not None:
            start_atom_idx = np.random.choice(mol.GetNumAtoms(), p=atom_weights)
        else:
            start_atom_idx = np.random.randint(0, mol.GetNumAtoms())
    elif start_atom_idx < 0 or start_atom_idx >= mol.GetNumAtoms():
        raise ValueError(f"start_atom_idx {start_atom_idx} out of range [0, {mol.GetNumAtoms()-1}]")
    
    # Initialize subgraph with starting atom
    subgraph_atoms = {start_atom_idx}
    candidate_neighbors = {}  # dict: neighbor_idx -> weight
    
    # Add initial neighbors to candidate set with weights
    start_atom = mol.GetAtomWithIdx(start_atom_idx)
    for neighbor in start_atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx not in subgraph_atoms:
            weight = 1.0
            if atom_weights is not None:
                weight *= atom_weights[neighbor_idx]
            if bond_weights is not None:
                bond_key = (start_atom_idx, neighbor_idx)
                bond_key_reverse = (neighbor_idx, start_atom_idx)
                if bond_key in bond_weights:
                    weight *= bond_weights[bond_key]
                elif bond_key_reverse in bond_weights:
                    weight *= bond_weights[bond_key_reverse]
            candidate_neighbors[neighbor_idx] = weight
    
    # Grow subgraph with weighted selection
    while len(subgraph_atoms) < min_size or (len(candidate_neighbors) > 0 and len(subgraph_atoms) < max_size):
        if not candidate_neighbors:
            break
        
        # Convert weights to probabilities
        neighbor_indices = list(candidate_neighbors.keys())
        weights = np.array(list(candidate_neighbors.values()))
        probabilities = weights / np.sum(weights)
        
        # Weighted random selection
        selected_neighbor = np.random.choice(neighbor_indices, p=probabilities)
        subgraph_atoms.add(selected_neighbor)
        del candidate_neighbors[selected_neighbor]
        
        # Add new neighbors to candidate set with weights
        selected_atom = mol.GetAtomWithIdx(selected_neighbor)
        for neighbor in selected_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in subgraph_atoms and neighbor_idx not in candidate_neighbors:
                weight = 1.0
                if atom_weights is not None:
                    weight *= atom_weights[neighbor_idx]
                if bond_weights is not None:
                    bond_key = (selected_neighbor, neighbor_idx)
                    bond_key_reverse = (neighbor_idx, selected_neighbor)
                    if bond_key in bond_weights:
                        weight *= bond_weights[bond_key]
                    elif bond_key_reverse in bond_weights:
                        weight *= bond_weights[bond_key_reverse]
                candidate_neighbors[neighbor_idx] = weight
    
    # Convert to list of integers and create subgraph
    atom_indices = list(subgraph_atoms)
    return get_rdkit_subgraph_by_atom_indices(mol, atom_indices)


def remove_aromaticity_from_atoms(mol: Chem.Mol, atom_indices: Union[List[int], np.ndarray]) -> Chem.Mol:
    """
    Remove aromaticity from specified atoms in an RDKit molecule.
    
    This function modifies the aromaticity flags of the specified atoms and their
    associated bonds, converting aromatic bonds to single bonds and removing
    aromatic atom flags.
    
    Args:
        mol: RDKit molecule object
        atom_indices: List or array of atom indices to remove aromaticity from
        
    Returns:
        RDKit molecule object with aromaticity removed from specified atoms
        
    Raises:
        ValueError: If atom indices are invalid
    """
    if isinstance(atom_indices, np.ndarray):
        atom_indices = atom_indices.tolist()
    
    if not atom_indices:
        return mol
    
    # Validate atom indices
    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range [0, {num_atoms})")
    
    # Create a copy of the molecule to modify
    rw_mol = Chem.RWMol(mol)
    
    # Remove aromaticity from specified atoms
    for atom_idx in atom_indices:
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        
        # Remove aromatic flag from atom
        atom.SetIsAromatic(False)
        
        # Remove aromaticity from bonds connected to this atom
        for bond in atom.GetBonds():
            bond_idx = bond.GetIdx()
            bond_obj = rw_mol.GetBondWithIdx(bond_idx)
            
            # If the bond is aromatic, convert to single bond
            if bond_obj.GetIsAromatic():
                bond_obj.SetIsAromatic(False)
                bond_obj.SetBondType(Chem.BondType.SINGLE)
    
    # Convert back to molecule and sanitize
    new_mol = rw_mol.GetMol()
    
    try:
        Chem.SanitizeMol(new_mol)
    except Exception as e:
        print(f"Warning: Sanitization failed: {e}")
    
    return new_mol


def remove_aromaticity_from_subgraph(mol: Chem.Mol, subgraph_mol: Chem.Mol) -> Chem.Mol:
    """
    Remove aromaticity from atoms that are part of a subgraph.
    
    This function identifies which atoms in the original molecule correspond to
    the subgraph and removes their aromaticity.
    
    Args:
        mol: Original RDKit molecule object
        subgraph_mol: RDKit molecule object representing the subgraph
        
    Returns:
        RDKit molecule object with aromaticity removed from subgraph atoms
        
    Raises:
        ValueError: If subgraph is not a valid substructure of the original molecule
    """
    # Find the subgraph match in the original molecule
    match = mol.GetSubstructMatch(subgraph_mol)
    if not match:
        raise ValueError("Subgraph is not a substructure of the original molecule")
    
    # Get atom indices from the match
    atom_indices = list(match)
    
    return remove_aromaticity_from_atoms(mol, atom_indices)


def remove_all_aromaticity(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove all aromaticity from an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        RDKit molecule object with all aromaticity removed
    """
    rw_mol = Chem.RWMol(mol)
    
    # Remove aromaticity from all atoms
    for atom in rw_mol.GetAtoms():
        atom.SetIsAromatic(False)
    
    # Remove aromaticity from all bonds
    for bond in rw_mol.GetBonds():
        if bond.GetIsAromatic():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
    
    # Convert back to molecule and sanitize
    new_mol = rw_mol.GetMol()
    
    try:
        Chem.SanitizeMol(new_mol)
    except Exception as e:
        print(f"Warning: Sanitization failed: {e}")
    
    return new_mol


def preserve_aromaticity_in_subgraph(mol: Chem.Mol, atom_indices: Union[List[int], np.ndarray]) -> Chem.Mol:
    """
    Remove aromaticity from all atoms except those specified in the subgraph.
    
    This function removes aromaticity from atoms that are NOT in the specified
    subgraph, preserving aromaticity only in the subgraph atoms.
    
    Args:
        mol: RDKit molecule object
        atom_indices: List or array of atom indices to preserve aromaticity for
        
    Returns:
        RDKit molecule object with aromaticity preserved only in subgraph atoms
        
    Raises:
        ValueError: If atom indices are invalid
    """
    if isinstance(atom_indices, np.ndarray):
        atom_indices = atom_indices.tolist()
    
    # Validate atom indices
    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            raise ValueError(f"Atom index {idx} out of range [0, {num_atoms})")
    
    # Get all atom indices
    all_atoms = set(range(num_atoms))
    subgraph_atoms = set(atom_indices)
    non_subgraph_atoms = list(all_atoms - subgraph_atoms)
    
    # Remove aromaticity from non-subgraph atoms
    return remove_aromaticity_from_atoms(mol, non_subgraph_atoms)


def get_aromatic_atoms(mol: Chem.Mol) -> List[int]:
    """
    Get indices of all aromatic atoms in an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of atom indices that are aromatic
    """
    aromatic_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atoms.append(atom.GetIdx())
    
    return aromatic_atoms


def get_aromatic_bonds(mol: Chem.Mol) -> List[int]:
    """
    Get indices of all aromatic bonds in an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of bond indices that are aromatic
    """
    aromatic_bonds = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            aromatic_bonds.append(bond.GetIdx())
    
    return aromatic_bonds


def analyze_aromaticity(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Analyze the aromaticity in an RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary containing aromaticity analysis results
    """
    aromatic_atoms = get_aromatic_atoms(mol)
    aromatic_bonds = get_aromatic_bonds(mol)
    
    # Get aromatic rings
    ring_info = mol.GetRingInfo()
    aromatic_rings = []
    
    for ring_atoms in ring_info.AtomRings():
        # Check if all atoms in the ring are aromatic
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atoms):
            aromatic_rings.append(list(ring_atoms))
    
    analysis = {
        'num_aromatic_atoms': len(aromatic_atoms),
        'num_aromatic_bonds': len(aromatic_bonds),
        'num_aromatic_rings': len(aromatic_rings),
        'aromatic_atom_indices': aromatic_atoms,
        'aromatic_bond_indices': aromatic_bonds,
        'aromatic_ring_indices': aromatic_rings,
        'aromatic_atom_symbols': [mol.GetAtomWithIdx(idx).GetSymbol() for idx in aromatic_atoms],
        'total_atoms': mol.GetNumAtoms(),
        'total_bonds': mol.GetNumBonds(),
        'aromaticity_ratio': len(aromatic_atoms) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
    }
    
    return analysis


def parse_scaffold(
    ligand: Union[str, Chem.Mol],
    generic: bool = True,
) -> Dict[str, Any]:
    """
    Returns Bemis–Murcko scaffold and (optionally) its generic version.
    Output includes SMILES and atom index mapping back to the original ligand.
    """
    mol = _to_mol(ligand)

    # 1) Bemis–Murcko scaffold (rings + linkers; removes side chains)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # RDKit Mol
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)

    # 2) Generic scaffold (all atoms/bonds genericized)
    generic_mol = MurckoScaffold.MakeScaffoldGeneric(Chem.Mol(scaffold)) if generic else None
    generic_smiles = Chem.MolToSmiles(generic_mol, canonical=True) if generic_mol else None

    # 3) Map scaffold atoms back to original ligand atom indices (first match)
    #    (A scaffold is a substructure of the original ligand.)
    match = mol.GetSubstructMatch(scaffold)  # tuple of atom indices in mol
    mapping = list(match) if match else []

    # 4) Some handy counts
    # n_rings = Chem.GetSSSR(scaffold)
    n_atoms_core = scaffold.GetNumAtoms()

    return {
        "scaffold_mol": scaffold,
        "scaffold_smiles": scaffold_smiles,
        "generic_scaffold_mol": generic_mol,
        "generic_scaffold_smiles": generic_smiles,
        "core_atom_mapping": mapping,  # ligand atom idx for each atom in scaffold (order = scaffold atom order)
        # "n_rings": int(n_rings),
        "n_atoms_core": n_atoms_core,
    }



from rdkit import Chem
from rdkit.Chem import rdmolops
import random
from collections import deque
from typing import Tuple, List, Set, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
import random
from collections import deque
from typing import Tuple, List, Set, Optional
from rdkit.Geometry import Point3D

def random_ring_substructure(
    mol: Chem.Mol,
    radius: int = 2,
    seed: Optional[int] = None,
    prefer_larger_rings: bool = False,
    keep_props: bool = True,
    copy_conformer: bool = True,
) -> Tuple[Chem.Mol, List[int], Set[int]]:
    """
    Pick a random ring, collect atoms within 'radius' bonds of that ring,
    and return the induced substructure.

    Returns:
      submol: new RDKit Mol containing the subgraph
      sub_to_parent: list s.t. submol atom i -> original atom index
      atom_set: set of original atom indices included
    """
    if seed is not None:
        random.seed(seed)

    # --- 1) pick a ring ---
    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    if not atom_rings:
        raise ValueError("No rings found in molecule.")

    ring = (random.choices(atom_rings, weights=[len(r) for r in atom_rings], k=1)[0]
            if prefer_larger_rings else random.choice(atom_rings))

    # --- 2) BFS to collect atoms within 'radius' (graph distance) ---
    atom_set: Set[int] = set(ring)
    dist = {a: 0 for a in ring}
    q = deque(ring)
    while q:
        a = q.popleft()
        if dist[a] == radius:
            continue
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nbr.GetIdx()
            if j not in dist:
                dist[j] = dist[a] + 1
                atom_set.add(j)
                q.append(j)

    atom_list = sorted(atom_set)

    # --- 3) build submol by copying selected atoms & bonds ---
    em = Chem.EditableMol(Chem.Mol())
    parent_to_sub = {idx: i for i, idx in enumerate(atom_list)}
    sub_to_parent = atom_list[:]  # submol atom i -> original atom index

    # add atoms
    for p_idx in atom_list:
        pa = mol.GetAtomWithIdx(p_idx)
        na = Chem.Atom(pa.GetAtomicNum())
        if keep_props:
            # copy basic props
            na.SetFormalCharge(pa.GetFormalCharge())
            na.SetIsAromatic(pa.GetIsAromatic())
            na.SetChiralTag(pa.GetChiralTag())
            na.SetNumExplicitHs(pa.GetNumExplicitHs())
            na.SetNoImplicit(pa.GetNoImplicit())
        em.AddAtom(na)

    # add bonds whose both endpoints are inside atom_set
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in atom_set and j in atom_set:
            bi = parent_to_sub[i]
            bj = parent_to_sub[j]
            bt = b.GetBondType()
            nb = em.AddBond(bi, bj, bt)
            # optional: copy aromaticity / stereochem
            # (RDKit will usually infer these on Sanitize)
    submol = em.GetMol()

    # sanitize (handles valence, aromaticity, etc.)
    Chem.SanitizeMol(submol)

    # --- 4) optionally copy 3D coordinates (first conformer) ---
    if copy_conformer and mol.GetNumConformers() > 0:
        conf_parent = mol.GetConformer()
        conf_sub = Chem.Conformer(submol.GetNumAtoms())
        for i_sub, i_parent in enumerate(sub_to_parent):
            p = conf_parent.GetAtomPosition(i_parent)
            conf_sub.SetAtomPosition(i_sub, Point3D(p.x, p.y, p.z))
        submol.RemoveAllConformers()
        submol.AddConformer(conf_sub, assignId=True)

    return submol, sub_to_parent, atom_set

from typing import Iterable, List, Tuple, Set, Union

def filter_by_atom_types(
    mol: Chem.Mol,
    allowed: Iterable[Union[str, int]],
    *,
    keep_largest_fragment: bool = True,
    copy_conformer: bool = True,
) -> Tuple[Chem.Mol, List[int]]:
    """
    Return a molecule containing only atoms whose element ∈ `allowed`.
    Bonds are kept if both endpoints survive.
    Also returns a mapping: filtered_atom_index -> original_atom_index.

    Args
    ----
    mol : RDKit Mol
    allowed : iterable of element symbols (e.g. ["C","N","O"]) or atomic numbers (e.g. [6,7,8])
    keep_largest_fragment : if True, keep only the largest connected component after filtering
    copy_conformer : if True and mol has conformers, copy coordinates to the filtered mol

    Returns
    -------
    filtered_mol : RDKit Mol
    mapping : list[int]
        mapping[i] = original atom index in `mol` for atom i in `filtered_mol`
    """
    # normalize allowed to atomic numbers
    pt = Chem.GetPeriodicTable()
    allowed_Z: Set[int] = {
        (pt.GetAtomicNumber(a) if isinstance(a, str) else int(a)) for a in allowed
    }

    # choose atoms to keep
    keep = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() in allowed_Z]
    if not keep:
        return Chem.Mol(), []

    # build new molecule
    em = Chem.EditableMol(Chem.Mol())
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep)}
    mapping = keep[:]  # new_idx -> old_idx

    # add atoms (copy a few key properties)
    for old_i in keep:
        a = mol.GetAtomWithIdx(old_i)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsAromatic(a.GetIsAromatic())
        na.SetChiralTag(a.GetChiralTag())
        na.SetNoImplicit(a.GetNoImplicit())
        na.SetNumExplicitHs(a.GetNumExplicitHs())
        em.AddAtom(na)

    # add bonds when both endpoints survive
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in old_to_new and j in old_to_new:
            em.AddBond(old_to_new[i], old_to_new[j], b.GetBondType())

    filtered = em.GetMol()
    Chem.SanitizeMol(filtered)

    # optionally keep only the largest fragment
    if keep_largest_fragment and filtered.GetNumAtoms() > 0:
        frags = Chem.GetMolFrags(filtered, asMols=False, sanitizeFrags=False)  # tuple of tuples of atom idx per frag
        # pick largest by atom count
        largest = max(frags, key=len)
        # remap to a new molecule consisting only of 'largest'
        sub_old_to_new = {old: i for i, old in enumerate(largest)}
        em2 = Chem.EditableMol(Chem.Mol())
        for old in largest:
            a = filtered.GetAtomWithIdx(old)
            na = Chem.Atom(a.GetAtomicNum())
            na.SetFormalCharge(a.GetFormalCharge())
            na.SetIsAromatic(a.GetIsAromatic())
            na.SetChiralTag(a.GetChiralTag())
            na.SetNoImplicit(a.GetNoImplicit())
            na.SetNumExplicitHs(a.GetNumExplicitHs())
            em2.AddAtom(na)
        for b in filtered.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in sub_old_to_new and j in sub_old_to_new:
                em2.AddBond(sub_old_to_new[i], sub_old_to_new[j], b.GetBondType())
        filtered2 = em2.GetMol()
        Chem.SanitizeMol(filtered2)

        # update mapping: new_idx -> original idx
        mapping = [mapping[old] for old in largest]
        filtered = filtered2

    # copy coordinates (if present)
    if copy_conformer and mol.GetNumConformers() > 0 and filtered.GetNumAtoms() > 0:
        conf_src = mol.GetConformer()
        conf_dst = Chem.Conformer(filtered.GetNumAtoms())
        for new_i, old_i in enumerate(mapping):
            p = conf_src.GetAtomPosition(old_i)
            conf_dst.SetAtomPosition(new_i, Point3D(p.x, p.y, p.z))
        filtered.RemoveAllConformers()
        filtered.AddConformer(conf_dst, assignId=True)

    return filtered, mapping

import pandas as pd
from typing import Dict, Any, Iterable, Optional

def nested_to_df(
    nested: Dict[Any, Dict[str, Any]],
    index_name: Optional[str] = None,
    columns_order: Optional[Iterable[str]] = None,
    fill_value: Any = None,
    sort_index: bool = False,
    sort_columns: bool = False,
) -> pd.DataFrame:
    """Build a DataFrame where first-level keys become the index and
    second-level keys are metric names (columns)."""
    df = pd.DataFrame.from_dict(nested, orient="index")

    if index_name is not None:
        df.index.name = index_name
    if columns_order is not None:
        cols = [c for c in columns_order if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]
    if fill_value is not None:
        df = df.fillna(fill_value)
    if sort_index:
        df = df.sort_index()
    if sort_columns:
        df = df.reindex(sorted(df.columns), axis=1)
    return df