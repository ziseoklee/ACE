# ACE for flexible-pose scaffold decoration

This project implements the paper's molecular application: composing pretrained de-novo (EDM), conformer (GeoDiff), and pocket-conditioned structure-based design (DiffSBDD) experts for flexible-pose scaffold decoration. ACE applies the paper's time-varying exponent correction and particle resampling to the heterogeneous ratio-of-densities path.

## Requirements

- Linux x86-64 with an NVIDIA CUDA GPU (the locked environment uses PyTorch 2.4.1 and CUDA 12.1 wheels)
- Git, curl, wget, and tar
- [uv](https://docs.astral.sh/uv/) and Python 3.11
- Approximately 15 GB of free space for environments, submodules, and model checkpoints, plus space for generated samples

QuickVina 2 is bundled at `src/lib/qvina2/qvina02`. Pretrained repositories and checkpoints retain their upstream licenses and terms.

## Installation

From a fresh clone:

```bash
git clone https://github.com/ziseoklee/ACE.git
cd ACE/molecule
bash scripts/setup_project.sh
```

The idempotent setup script initializes the three pinned Git submodules, applies the checked-in packaging/runtime compatibility patches, downloads the DiffSBDD and GeoDiff checkpoints when absent, creates `molecule/.venv` with `uv venv`, and installs the committed environment with `uv sync --frozen`.

Run the setup from inside the ACE repository. Model downloads require network access; rerunning the script skips checkpoints already present.

Verify the installation without sampling:

```bash
uv run ace-infer --help
uv run ace-evaluate --help
uv run python -m compileall -q src
```

## Single-condition ACE inference

The following command uses the paper's four-expert scaffold-decoration composition and ACE bump parameters (`B1=30`, `B2=0.336`) on the included `4m7t` example:

```bash
uv run ace-infer \
  sampler=ACESampler \
  sampler.batch_size=5 \
  sampler.seed=42 \
  moe.omega=1.4 \
  moe.diffusion_scale=2.0 \
  moe.exponents.diffsbdd.weight_fn.B1=30 \
  moe.exponents.diffsbdd.weight_fn.B2=0.336 \
  data.num_ligand_atoms=28
```

The default inputs are `examples/4m7t_pocket.pdb`, `examples/4m7t_fragment.sdf`, and `examples/4m7t_ligand.sdf`. Set `data.num_ligand_atoms=null` to use the reference ligand atom count, or set an integer to control the generated ligand size. Hydra writes the resolved config and generated SDF/XYZ/PNG files under `outputs/` unless `output_dir` is overridden.

Inspect the complete resolved configuration with:

```bash
uv run ace-infer --cfg job moe.omega=1.4
```

Configuration definitions are in `src/configs/`; `inference.yaml` records the default paper composition. `moe.diffusion_scale=2.0` is an empirical setting for DiffSBDD and four-expert MoE inference and should be reported when comparing configurations. It is not consistently optimal for single-expert EDM or GeoDiff runs.

For controlled examples covering each single expert, two-expert guidance, and the four-expert ACE composition, run:

```bash
bash scripts/run_example_moe_sampling.sh
```

The script accepts environment overrides such as `DEVICE=cuda:1`, `OMEGA=1.4`, `SEED=42`, `B1=30`, and `B2=0.336`.

## CrossDocked2020 benchmark

The repository includes the 76 processed benchmark tasks and their corresponding pocket/ligand files under `data/crossdocked/`. They are derived from the processed CrossDocked2020 release used by [Delete](https://www.nature.com/articles/s42256-025-00997-w), available from [Zenodo record 7980002](https://zenodo.org/records/7980002).

Generate five samples per task (one trial with the default batch size of five) using the paper configuration:

```bash
uv run ace-crossdocked-infer \
  benchmark.num_trials=1 \
  benchmark.seed=42 \
  sampler=ACESampler \
  sampler.batch_size=5 \
  moe.omega=1.4 \
  moe.diffusion_scale=2.0 \
  moe.exponents.diffsbdd.weight_fn.B1=30 \
  moe.exponents.diffsbdd.weight_fn.B2=0.336
```

The run directory has this structure:

```text
outputs/crossdocked2020/<run>/
├── inference/<task_id>/<sample>.sdf
├── inference/<task_id>/<sample>.xyz
├── inference/<task_id>/<sample>.png
└── inference/<task_id>/fragment.png
```

Evaluate druglikeness and QuickVina 2 docking for a completed run:

```bash
uv run ace-evaluate crossdocked \
  --run_dir outputs/crossdocked2020/<run> \
  --metrics druglikeness \
  --metrics docking \
  --expected_num_samples 5
```

This writes sample-, task-, and run-level tables to `<run>/evaluation/{samples,tasks,summary}.csv`. Keep the inference seed, batch size, guidance scale, bump parameters, diffusion scale, docking seed, and docking settings unchanged when comparing with the paper.

## Evaluate an individual molecule

The included examples provide lightweight evaluation checks:

```bash
uv run ace-evaluate druglikeness \
  --ligand_sdf examples/4m7t_ligand.sdf

uv run ace-evaluate docking \
  --pocket_pdb examples/4m7t_pocket.pdb \
  --ligand_sdf examples/4m7t_ligand.sdf \
  --ref_ligand_sdf examples/4m7t_ligand.sdf \
  --seed 42
```

Use `uv run ace-evaluate <command> --help` for all metric and output options.

## Implementation map

```text
src/
├── run_inference.py              # One scaffold-decoration condition
├── run_crossdocked_inference.py  # CrossDocked2020 benchmark driver
├── configs/                      # Hydra configs and paper parameters
├── experts/                      # EDM, GeoDiff, and DiffSBDD adapters
├── pipelines/                    # Component specifications and schedulers
├── sampling/                     # Probability paths, ACE weights, and samplers
├── postprocessing/               # Point-cloud-to-molecule construction
├── evaluation/                   # Druglikeness, docking, and summaries
├── pretrained_models/            # Pinned upstream Git submodules
└── utils/
```

The shared atom state is `[x, y, z, atom_type..., optional_nuclear_charge_feature]`. `src/sampling/moe_layout.py` constructs a canonical union of the experts' atom vocabularies and adapts it back to each expert's native feature order. Generated point clouds are converted through an XYZ block, Open Babel bond-order inference, and a fresh RDKit molecule. Because scaffold topology is not forcibly restored during postprocessing, preservation should be evaluated from the generated molecule topology.

To inspect processed task 85 and write fragment/ligand topology images without running inference:

```bash
uv run python src/utils/peek_crossdocked_sample.py 85
```

## Reproducibility notes

- The submodule commits are pinned in the parent repository; do not update them for paper reproduction.
- The setup patches are idempotent but modify the checked-out submodule worktrees locally. This is expected.
- CUDA kernels, Open Babel bond inference, and docking can introduce small platform-dependent differences.
- Full benchmark inference and docking are expensive. Confirm one example first, then retain the resolved Hydra config stored with each run.
