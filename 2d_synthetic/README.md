# ACE synthetic checker experiments

This project reproduces the paper's synthetic checker benchmark: the path-existence criterion, Marginal Path Collapse under heterogeneous schedules, and ACE's endpoint-preserving correction. The code trains the three experts in the paper's heterogeneous factorization, evaluates NR/FKC/ACE, and generates the synthetic figures and appendix analyses.

## Requirements

- Python 3.11 managed by [uv](https://docs.astral.sh/uv/)
- An NVIDIA CUDA GPU for the full evaluation (`ace_eval_script.py` uses 10,000 particles and the paper configuration)
- Approximately 30 minutes on an RTX A6000 to train the toy experts; total evaluation time depends on the requested sweep

All commands below start in this directory:

```bash
cd ACE/2d_synthetic
uv venv --python 3.11
uv sync --frozen
```

`uv sync` installs the locked CUDA 12.4 PyTorch build into `2d_synthetic/.venv`. Use `uv run` for every command; activating the environment is optional.

## 1. Train the toy experts

```bash
uv run python ace_lib/train_toy_models.py
```

The script uses seed `0` and trains every schedule in `ace_lib/interpolant_schedules.json`. Checkpoints are written to `PretrainedToyModels/`, with separate velocity and score networks for the local conditional, global conditional, and marginal experts.

## 2. Reproduce the quantitative sweep

```bash
uv run python ace_eval_script.py
```

The script evaluates five seeds for NR, FKC, and the ACE bump/ramp sweep. It writes sample diagnostics, `results_checker_NR_FKC_ACE_multiple_configs.csv`, and `stats_summary_NR_FKC_ACE_multiple_configs.csv` under a timestamped `ace_eval_runs_YYYYMMDD/` directory.

This is the full experiment rather than a smoke test: it uses 10,000 particles, 1,000 integration steps, and all configured bump/ramp combinations. Train the checkpoints first and run it from this directory so relative paths resolve correctly.

## 3. Reproduce figures and appendix analyses

Launch the checked-in notebook from the project environment:

```bash
uv run jupyter lab ace_demo.ipynb
```

Run the notebook sections in order. The sections are labeled for the main synthetic figures and tables and for the appendix schedule, bump, and ESS analyses. Outputs are written under `ace_demo_runs_YYYYMMDD/`. The notebook expects the checkpoints in `PretrainedToyModels/`.

## Validation

The following checks imports and syntax without running the expensive experiments:

```bash
uv run python -m compileall -q ace_lib ace_eval_script.py
uv run python -c "import torch, ot, pandas, scipy; import ace_lib; print(torch.__version__)"
```

Numerical results can vary slightly across GPU architectures and CUDA kernels. Use the committed `uv.lock`, unchanged seeds, schedules, particle count, and integration steps for the closest comparison with the paper.
