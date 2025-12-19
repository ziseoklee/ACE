from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rdkit import Chem
from torch.utils.data import Dataset
import pickle
import sys
import lmdb
import torch


PathLike = Union[str, Path]


class CrossDockDataset(Dataset):
    """
    A lightweight dataset for CrossDock-style protein–ligand pairs.

    This class scans a root directory for pairs of:
      - protein PDB file (protein structure)
      - ligand file (reference ligand), supported: .sdf, .mol2, .mol, .pdb

    By default, it pairs within directories: if a directory contains at least
    one .pdb and one ligand file, each protein is paired with the first ligand
    discovered in that directory.

    __getitem__ returns a dictionary with:
      - "protein_pdb_path": str path to the protein PDB file
      - "reference_ligand_mol": RDKit Mol object for the ligand

    Parameters
    ----------
    root_dir : str | Path
        Root directory containing the processed CrossDock data.
    pair_by : str
        Pairing strategy. Options:
          - "directory" (default): pair first ligand with each protein within the same directory
          - "stem": pair ligand and protein if they share the same filename stem
    recursive : bool
        Whether to recursively scan subdirectories. Default: True.
    ligand_extensions : Sequence[str]
        Allowed ligand file extensions, in preference order. Default: (".sdf", ".mol2", ".mol", ".pdb").
    sort : bool
        Sort discovered samples for deterministic ordering. Default: True.
    max_samples : Optional[int]
        If provided, truncate the dataset to the first N samples.
    """

    def __init__(
        self,
        root_dir: PathLike,
        pair_by: str = "directory",
        recursive: bool = True,
        ligand_extensions: Sequence[str] = (".sdf", ".mol2", ".mol", ".pdb"),
        sort: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.pair_by = pair_b

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        protein_path, ligand_path = self.samples[index]
        ligand_mol = self._load_ligand(ligand_path)

        return {
            "protein_pdb_path": str(protein_path),
            "reference_ligand_mol": ligand_mol,
            'scaffold_mol': scaffold_mol,
            'scaffold_smiles': scaffold_smiles,
        }

    # --- Pairing strategies -------------------------------------------------
    def _find_pairs_by_directory(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        walker = os.walk(self.root_dir) if self.recursive else [(self.root_dir, [], os.listdir(self.root_dir))]

        for dirpath, _dirnames, filenames in walker:
            if not filenames:
                continue

            directory = Path(dirpath)
            pdb_files: List[Path] = [directory / f for f in filenames if f.lower().endswith(".pdb")]

            # Prefer non-PDB ligand formats first, then fall back to .pdb
            ligand_files: List[Path] = []
            for ext in self.ligand_extensions:
                ligand_files.extend([directory / f for f in filenames if f.lower().endswith(ext)])

            # Remove any protein file from ligand candidates if identical path
            ligand_files = [lf for lf in ligand_files if lf not in pdb_files]

            if not pdb_files or not ligand_files:
                continue

            # Pair each protein in the folder with the first ligand discovered
            first_ligand = ligand_files[0]
            for protein in pdb_files:
                pairs.append((protein, first_ligand))

        return pairs

    def _find_pairs_by_stem(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []

        pdb_files = list(self.root_dir.rglob("*.pdb") if self.recursive else self.root_dir.glob("*.pdb"))
        ligand_files: List[Path] = []
        for ext in self.ligand_extensions:
            if self.recursive:
                ligand_files.extend(self.root_dir.rglob(f"*{ext}"))
            else:
                ligand_files.extend(self.root_dir.glob(f"*{ext}"))

        # Map stems to files for quick lookups
        stem_to_pdb: Dict[str, List[Path]] = {}
        for p in pdb_files:
            stem_to_pdb.setdefault(p.stem.lower(), []).append(p)

        stem_to_lig: Dict[str, List[Path]] = {}
        for l in ligand_files:
            stem_to_lig.setdefault(l.stem.lower(), []).append(l)

        # Intersect stems; pair each protein with the first ligand sharing the stem
        for stem, proteins in stem_to_pdb.items():
            cand_ligs = stem_to_lig.get(stem, [])
            if not cand_ligs:
                continue
            first_ligand = cand_ligs[0]
            for protein in proteins:
                if protein != first_ligand:
                    pairs.append((protein, first_ligand))

        return pairs

    # --- Ligand loading -----------------------------------------------------
    def _load_ligand(self, ligand_path: Path) -> Chem.Mol:
        ext = ligand_path.suffix.lower()

        if ext == ".sdf":
            suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(str(ligand_path), sanitize=True, removeHs=False)
        elif ext == ".mol":
            mol = Chem.MolFromMolFile(str(ligand_path), sanitize=True, removeHs=False)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(str(ligand_path), sanitize=True, removeHs=False)
        else:
            raise ValueError(f"Unsupported ligand extension: {ext} ({ligand_path})")

        if mol is None:
            raise ValueError(f"Failed to load RDKit Mol from: {ligand_path}")

        return mol


__all__ = ["CrossDockDataset"]


class CrossDockLMDBDataset(Dataset):
    """
    LMDB-backed CrossDock dataset for returning protein PDB path and RDKit ligand Mol.

    The LMDB at `lmdb_path` stores pickled `ProteinLigandData` objects (from
    `baseline/Delete/utils/data.py`). Each entry typically includes:

      - ligand_mol / mol: RDKit Mol
      - protein_filename: relative path to pocket PDB (e.g., '.../xxx_pocket10.pdb')

    Optionally uses `crossdocked_pocket10_molname2id.pt` to map
    (pocket_pdb_rel, ligand_sdf_rel) -> integer id, and `split_by_name.pt` to
    select a split ('train' or 'test').

    __getitem__ returns dict:
      - "protein_pdb_path": str (protein_root / protein_filename)
      - "reference_ligand_mol": RDKit Mol

    Parameters
    ----------
    lmdb_path : str | Path
        Path to crossdocked_pocket10_mol.lmdb
    name2id_path : Optional[str | Path]
        Path to crossdocked_pocket10_molname2id.pt (required if using splits)
    split_path : Optional[str | Path]
        Path to split_by_name.pt (required if using splits)
    split : Optional[str]
        If provided, one of {"train", "test"} to select subset using split_by_name.pt
    protein_root : Optional[str | Path]
        Base directory to resolve protein_filename into a full PDB path
    ensure_utils_data : bool
        Try to import 'utils.data' by adding baseline/Delete to sys.path for unpickling
    sort : bool
        Sort samples for deterministic ordering (default: True)
    max_samples : Optional[int]
        If provided, truncate dataset to first N samples
    strict_protein_path : bool
        If True, raise FileNotFoundError when resolved protein PDB path doesn't exist
    """

    def __init__(
        self,
        lmdb_path: PathLike,
        name2id_path: Optional[PathLike] = None,
        split_path: Optional[PathLike] = None,
        split: Optional[str] = None,
        protein_root: Optional[PathLike] = None,
        ensure_utils_data: bool = True,
        sort: bool = True,
        max_samples: Optional[int] = None,
        strict_protein_path: bool = False,
    ) -> None:
        self.lmdb_path = Path(lmdb_path).expanduser().resolve()
        self.name2id_path = Path(name2id_path).expanduser().resolve() if name2id_path is not None else None
        self.split_path = Path(split_path).expanduser().resolve() if split_path is not None else None
        self.split = split
        self.protein_root = Path(protein_root).expanduser().resolve() if protein_root is not None else None
        self.ensure_utils_data = ensure_utils_data
        self.sort = sort
        self.max_samples = max_samples
        self.strict_protein_path = strict_protein_path

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {self.lmdb_path}")

        if self.ensure_utils_data:
            self._ensure_utils_data_import()

        self.env = lmdb.open(
            str(self.lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Determine index -> LMDB key mapping
        if self.split is not None:
            if self.name2id_path is None or self.split_path is None:
                raise ValueError("name2id_path and split_path are required when 'split' is provided")
            self.index_to_id = self._build_index_from_split(self.name2id_path, self.split_path, self.split)
        else:
            # Fallback: iterate all integer keys using name2id if provided, otherwise LMDB stat
            if self.name2id_path is not None:
                name2id = torch.load(self.name2id_path, map_location='cpu')
                ids = list(name2id.values())
            else:
                with self.env.begin() as txn:
                    stat = txn.stat()
                    entries = stat.get('entries', 0)
                ids = list(range(entries))
            self.index_to_id = ids

        if self.sort and isinstance(self.index_to_id, list):
            self.index_to_id = sorted(self.index_to_id)

        if self.max_samples is not None:
            self.index_to_id = self.index_to_id[: self.max_samples]

        if len(self.index_to_id) == 0:
            raise RuntimeError("No samples found for LMDB dataset")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.index_to_id)

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        sample_id = self.index_to_id[index]
        key = str(sample_id).encode('ascii')
        with self.env.begin() as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"LMDB key not found: {sample_id}")

        data = pickle.loads(value)

        ligand_mol = None
        protein_pdb_rel = None
        if hasattr(data, 'ligand_mol') and data.ligand_mol is not None:
            ligand_mol = data.ligand_mol
        elif hasattr(data, 'mol') and data.mol is not None:
            ligand_mol = data.mol
        else:
            ligand_mol = self._build_rdkit_mol_from_fields(data)

        if hasattr(data, 'protein_filename'):
            protein_pdb_rel = data.protein_filename

        protein_pdb_path = None
        if protein_pdb_rel is not None and self.protein_root is not None:
            protein_pdb_path = (self.protein_root / protein_pdb_rel).as_posix()
            if self.strict_protein_path and not Path(protein_pdb_path).exists():
                raise FileNotFoundError(f"Protein PDB path does not exist: {protein_pdb_path}")
        else:
            # Fall back to the relative name if no root is provided
            protein_pdb_path = str(protein_pdb_rel) if protein_pdb_rel is not None else None

        return {
            "protein_pdb_path": protein_pdb_path,
            "reference_ligand_mol": ligand_mol,
        }

    # ------------------------------------------------------------------
    def _build_index_from_split(self, name2id_path: Path, split_path: Path, split: str) -> List[int]:
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be one of {'train','test'}")
        name2id = torch.load(name2id_path, map_location='cpu')
        split_map = torch.load(split_path, map_location='cpu')
        pairs = list(split_map[split])
        ids: List[int] = []
        missing = 0
        for pair in pairs:
            idx = name2id.get(pair, None)
            if idx is None:
                missing += 1
                continue
            ids.append(int(idx))
        if missing > 0 and len(ids) == 0:
            raise RuntimeError("No matching ids found from split pairs in name2id mapping")
        return ids

    def _ensure_utils_data_import(self) -> None:
        try:
            import utils.data  # noqa: F401
            return
        except Exception:
            pass
        # Attempt to add baseline/Delete to path
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / 'baseline' / 'Delete'
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            import utils.data  # noqa: F401
        except Exception as e:
            raise ImportError(
                f"Failed to import 'utils.data' needed to unpickle LMDB entries.\n"
                f"Tried adding {candidate} to sys.path. Original error: {e}"
            )

    def _build_rdkit_mol_from_fields(self, data: object) -> Chem.Mol:
        # Fallback construction using ligand_* fields
        if not (hasattr(data, 'ligand_element') and hasattr(data, 'ligand_bond_index') and hasattr(data, 'ligand_bond_type')):
            raise ValueError("Ligand fields not sufficient to reconstruct RDKit Mol")
        atomic_numbers = torch.as_tensor(getattr(data, 'ligand_element')).view(-1).tolist()
        bond_index = torch.as_tensor(getattr(data, 'ligand_bond_index')).cpu().numpy()
        bond_type_vec = torch.as_tensor(getattr(data, 'ligand_bond_type')).view(-1).cpu().numpy()

        rwmol = Chem.RWMol()
        atom_indices: List[int] = []
        for z in atomic_numbers:
            atom = Chem.Atom(int(z))
            atom_indices.append(rwmol.AddAtom(atom))

        def map_bond_type(t: int) -> Chem.BondType:
            if t == 1:
                return Chem.BondType.SINGLE
            if t == 2:
                return Chem.BondType.DOUBLE
            if t == 3:
                return Chem.BondType.TRIPLE
            # fallback
            return Chem.BondType.SINGLE

        for a, b, bt in zip(bond_index[0].tolist(), bond_index[1].tolist(), bond_type_vec.tolist()):
            if a == b:
                continue
            bt_enum = map_bond_type(int(bt))
            try:
                rwmol.AddBond(int(a), int(b), bt_enum)
            except Exception:
                # Avoid duplicate bonds if present
                pass
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

        # Set 3D coordinates if available
        if hasattr(data, 'ligand_pos'):
            pos = torch.as_tensor(getattr(data, 'ligand_pos')).cpu().numpy()
            if pos.ndim == 2 and pos.shape[0] == mol.GetNumAtoms() and pos.shape[1] == 3:
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i, (x, y, z) in enumerate(pos.tolist()):
                    conf.SetAtomPosition(i, (float(x), float(y), float(z)))
                mol.AddConformer(conf, assignId=True)
        return mol


__all__.append("CrossDockLMDBDataset")


