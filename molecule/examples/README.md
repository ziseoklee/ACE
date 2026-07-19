# Molecular example inputs

This directory contains three ready-to-run scaffold-decoration conditions: `3nfb`, `4m7t`, and `4yhj`. Each prefix has:

- `<prefix>_pocket.pdb`: protein-pocket coordinates
- `<prefix>_fragment.sdf`: input scaffold/fragment
- `<prefix>_ligand.sdf`: reference ligand used for metadata, atom-count selection, or docking-box placement
- `<prefix>_fragment.png` and `<prefix>_ligand.png`: 2D previews

The default inference config uses `4m7t`. To sample another condition from `molecule/`, override all three paths together:

```bash
uv run ace-infer \
  sampler=ACESampler \
  moe.omega=1.4 \
  data.protein_pocket_pdb_path=examples/4yhj_pocket.pdb \
  data.fragment_sdf_path=examples/4yhj_fragment.sdf \
  data.ligand_sdf_path=examples/4yhj_ligand.sdf \
  data.num_ligand_atoms=null
```

These files are extracted from the processed CrossDocked2020 examples included under `../data/crossdocked/`. See the parent [`README.md`](../README.md) for setup, paper parameters, output structure, and evaluation commands.
