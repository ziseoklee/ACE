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

For the HTTP API, [`inference-config.json`](inference-config.json) provides a complete request config: ten particles, 500 sampling steps, seed 42, and a ligand atom count determined from the normalized reference ligand. With the [backend server](../backend/README.md) running, submit from the `molecule/` project root:

```bash
curl --fail-with-body http://localhost:8000/api/v1/inference/jobs \
  -F 'pocket_pdb=@examples/4m7t_pocket.pdb' \
  -F 'fragment_sdf=@examples/4m7t_fragment.sdf' \
  -F 'reference_ligand_sdf=@examples/4m7t_ligand.sdf' \
  -F 'config=<examples/inference-config.json'
```

`config=<...` sends the JSON file's contents as a regular form field. To use another condition, change all three molecular input paths together; the same JSON config can be reused.

The molecular input files are extracted from the processed CrossDocked2020 examples included under `../data/crossdocked/`. See the parent [`README.md`](../README.md) for setup, paper parameters, output structure, and evaluation commands.
