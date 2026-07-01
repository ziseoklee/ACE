# ACE for SBDD

Implementation of ACE for drug design tasks presented in the paper: scaffold decoration.

## Usage

### Setup project

```bash
git clone https://github.com/ziseoklee/ACE.git
cd ACE/molecule

bash scripts/setup_project.sh
```

### Run Mixture of Experts (MoE) sampling for scaffold decoration

You can run ACE sampler with the following command:

```bash
uv run ace-infer \
    sampler=ACESampler \
    sampler.batch_size=5 \
    sampler.seed=42 \
    weight=ACEBumpWeight \
    weight.omega=1.4 \
    weight.B1=30 \
    weight.B2=0.336
```

Please refer to [`config_sampler.py`](src/configs/config_sampler.py), [`config_weight.py`](src/configs/config_weight.py), [`config_benchmark.py`](src/configs/config_benchmark.py), [`inference.yaml`](src/configs/inference.yaml), and [`crossdocked_inference.yaml`](src/configs/crossdocked_inference.yaml) for configuration options. [`config.py`](src/configs/config.py) registers these structured configs with Hydra. You can also run `uv run ace-infer --cfg job` to print the resolved inference config.

### Run evaluation for generated samples

We support two evaluation metrics: druglikeness and docking. You can run the evaluation with the following commands:

```bash
uv run ace-evaluate druglikeness --ligand_sdf examples/4m7t_ligand.sdf

# Druglikeness evaluation will look like:
# {
#   "Lipinski": 0.8,
#   "LogP": -4.141399999999994,
#   "QED": 0.186774247976761,
#   "SA": 0.44362465057886624,
#   "ligand_sdf": "examples/4m7t_ligand.sdf",
#   "validity": 1.0
# }


uv run ace-evaluate docking \
    --pocket_pdb examples/4m7t_pocket.pdb \
    --ligand_sdf examples/4m7t_ligand.sdf \
    --seed 42

# Docking evaluation output will look like:
# {
#   "docking_error": "",
#   "docking_success": 1.0,
#   "ligand_sdf": "examples/4m7t_ligand.sdf",
#   "pocket_pdb": "examples/4m7t_pocket.pdb",
#   "qvina_affinity": -8.6,
#   "qvina_cmd": "/path/to/project/src/lib/qvina2/qvina02 --receptor /tmp/tmps3i6_twt/receptor.pdbqt --ligand /tmp/tmps3i6_twt/ligand.pdbqt --center_x 4.010 --center_y -10.296 --center_z 17.053 --size_x 21.756 --size_y 21.869 --size_z 27.589 --exhaustiveness 8 --num_modes 9 --seed 42 --out /tmp/tmps3i6_twt/qvina_poses.pdbqt",
#   "qvina_num_poses": 9,
#   "ref_ligand_sdf": null
# }
```

Please check `uv run ace-evaluate druglikeness --help` or `uv run ace-evaluate docking --help` for more details on the evaluation options. Or you can check the cli entrypoint in `src/evaluation/cli.py` for more details.

### Run benchmark for CrossDocked2020

We use processed CrossDocked2020 test data of [Delete](https://www.nature.com/articles/s42256-025-00997-w) in https://zenodo.org/records/7980002. Excluding the case where the ligand has more than 29 atoms, we have 76 ligand-pocket pairs for evaluation. You can run the benchmark with the following command:

```bash
# Inference
uv run ace-crossdocked-infer \
    benchmark.num_trials=1 \
    benchmark.seed=42 \
    sampler=ACESampler \
    sampler.batch_size=5 \
    weight=ACEBumpWeight \
    weight.omega=1.4 \
    weight.B1=30 \
    weight.B2=0.336

# Inference output structure will look like:
#   outputs/crossdocked2020/{sampler_weight_timestamp}/inference/{task_id}/{sample}.sdf
#   outputs/crossdocked2020/{sampler_weight_timestamp}/inference/{task_id}/{sample}.xyz
#   outputs/crossdocked2020/{sampler_weight_timestamp}/inference/{task_id}/{sample}.png
#   outputs/crossdocked2020/{sampler_weight_timestamp}/inference/{task_id}/fragment.png


# Evaluation
uv run ace-evaluate crossdocked \
    --run_dir outputs/crossdocked2020/{run_name} \
    --metrics druglikeness --metrics docking \
    --expected_num_samples 5

# Evaluation output structure will look like:
#   outputs/crossdocked2020/{run}/evaluation/{samples,tasks,summary}.csv
```

## Source layout

The scaffold-decoration workflow is organized around a few small layers:

```text
src/
├── run_inference.py              # Hydra entrypoint for one scaffold-decoration inference
├── run_crossdocked_inference.py  # Hydra entrypoint for CrossDocked2020 benchmark inference
├── configs/                      # Hydra structured configs and default yaml files
├── inference/                    # Condition-level orchestration and shared runtime loading
├── experts/                      # Adapters for pretrained DiffSBDD, EDM, and GeoDiff experts
├── sampling/                     # Probability paths, schedulers, MoE path construction, and samplers
├── postprocessing/               # Molecule building plus deprecated scaffold/valence helpers
├── evaluation/                   # Click CLI, metrics, CrossDocked2020 evaluation, and summaries
├── pretrained_models/            # Vendored pretrained expert repositories
├── lib/                          # Packaged external binaries such as qvina2
└── utils/                        # Small shared utilities
```

The command-line entrypoints mirror this separation:

- `ace-infer`: single-condition scaffold-decoration inference
- `ace-crossdocked-infer`: CrossDocked2020 benchmark inference
- `ace-evaluate`: per-sample and CrossDocked2020 evaluation

Pretrained model code is kept under `src/pretrained_models/`, while `src/experts/` provides the ACE/MoE-facing adapter layer. The sampler treats generated ligands as point clouds. Because SDE-based sampling only produces atom coordinates and atom features, bond topology is assigned afterward in `src/postprocessing/`.

The current molecule-building flow is:

```text
xh point-cloud tensor -> XYZ block -> OpenBabel Python API bond-order guess -> fresh RDKit Mol copy
```

Scaffold topology is not enforced during this postprocessing step. Generated samples are first converted from point clouds into molecules, and fragment preservation should be evaluated afterward from the generated molecule topology.

### Utilities

To inspect a CrossDocked2020 processed sample without running inference, use:

```bash
uv run python src/utils/peek_crossdocked_sample.py 85
```

This prints the stored pocket/scaffold metadata and writes 2D topology PNGs for the fragment and ligand.

### Atom feature layout

The current shared MoE layout is specialized for the GeoDiff-QM9, EDM-QM9, and DiffSBDD-CrossDocked expert combination. If you swap or add experts, update `src/sampling/moe_layout.py` so the shared atom-feature vocabulary and each expert mask match the models' training representations. The layout file keeps model-specific constants such as `_DIFFSBDD_CROSSDOCKED_ACTIVE_ATOMS`, `_EDM_QM9_ACTIVE_ATOMS`, and their active/padding columns.

The shared per-atom state is:

```text
[x, y, z, C, N, O, S, B, Br, Cl, P, I, F, H, nuclear_charge_feature]
```

The atom-type indices are:

```text
C=0, N=1, O=2, S=3, B=4, Br=5, Cl=6, P=7, I=8, F=9, H=10
```

Expert representations differ:

- DiffSBDD-CrossDocked uses `C, N, O, S, B, Br, Cl, P, I, F, others`. In the current MoE combination, DiffSBDD owns the heavy-atom decode for `C/N/O/S/B/Br/Cl/P/I/F`; `H` and the nuclear-charge feature are padding for DiffSBDD.
- EDM-QM9 uses `H, C, N, O, F` plus an integer nuclear-charge feature. In the current MoE combination, only the EDM-owned `H` channel is decoded by EDM postprocessing; `C/N/O/F` are left to DiffSBDD to avoid double unnormalization. The final integer feature is the atomic number feature, not RDKit formal charge, and is ignored by molecule construction.
- GeoDiff uses fragment atom types as fixed auxiliary features for the fragment coordinate path. The nuclear-charge feature is zero-padded there.

## Additional Remarks

Combining DiffSBDD and EDM in the current MoE setup is relatively straightforward because both models apply the same `1/4` scaling to atom features, which makes their sample spaces reasonably aligned. More general expert combinations may require explicit feature-space calibration.
