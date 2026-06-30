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
├── postprocessing/               # Molecule building, scaffold topology enforcement, valence fixes
├── evaluation/                   # Click CLI, metrics, CrossDocked2020 evaluation, and summaries
├── pretrained_models/            # Vendored pretrained expert repositories
├── lib/                          # Packaged external binaries such as qvina2
└── utils/                        # Small shared utilities
```

The command-line entrypoints mirror this separation:

- `ace-infer`: single-condition scaffold-decoration inference
- `ace-crossdocked-infer`: CrossDocked2020 benchmark inference
- `ace-evaluate`: per-sample and CrossDocked2020 evaluation

Pretrained model code is kept under `src/pretrained_models/`, while `src/experts/` provides the ACE/MoE-facing adapter layer. The sampler treats generated ligands as point clouds. Converting those point clouds into RDKit molecules by adding bonds and enforcing scaffold topology is handled in `src/postprocessing/`.
