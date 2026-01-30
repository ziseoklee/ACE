# Molecule Experiments

Minimal instructions to reproduce the ACE pipelines. All commands assume you run from the current directory.

## Quickstart
- Optional: `conda env create -f environment.yml` then `conda activate substeer` (environment file at repo root).
- Ensure datasets exist at `data/crossdock/raw` and `data/crossdock/raw_sbdd`.
- Populate `src/pretrained_models/DiffSBDD/dataset` using the official DiffSBDD repository assets before running SBDD scripts.
- Place the official Delete repository contents under `baseline/Delete` so the baseline source files are available.
- Results are written under `result_inference/`, `result_evaluation/`, and `result_analysis/`.

## src layout
- `src/datasets/`: dataset loaders; `crossdock.py` builds crossdock variants.
- `src/distributions.py`: probability distribution utilities for training/inference.
- `src/hcg.py`: model/graph construction helpers (hierarchical coarse-graining pieces).
- `src/probability_path.py`: probability path / scheduling helpers for diffusion flows.
- `src/pretrained_models/`: bundled backbones and weights (`DiffSBDD/`, `GeoDiff/`, `e3_diffusion_for_molecules/`).

## Crossdock (the crossdock-small in paper)
1) Inference → ligands into `result_inference/crossdock_1.3_bump`
   ```
   bash scripts/crossdock.sh
   ```
2) Evaluation → metrics into `result_evaluation/crossdock/crossdock_1.3_bump*`
   ```
   bash scripts/evaluate.sh
   ```
3) Analysis → summarized scores into `result_analysis/`
   ```
   bash scripts/analyze_evaluation.sh
   ```

## Crossdock_SBDD
1) Inference → ligands into `result_inference/crossdock_sbdd_1.3_bump`
   ```
   bash scripts/crossdock_sbdd.sh
   ```
2) Evaluation → metrics into `result_evaluation/crossdock/crossdock_sbdd_1.3_bump*`
   ```
   bash scripts/evaluate_sbdd.sh
   ```
3) Analysis → summarized scores into `result_analysis/`
   ```
   bash scripts/analyze_evaluation_sbdd.sh
   ```

## Notes
- Scripts call `run.py` entrypoints such as `inference.crossdock`, `evaluation.evaluate_crossdock_qvina`, and `analysis.analyze_qvina_score(_sbdd)`.
- Adjust flags inside the scripts (e.g., `--inverse_temperature`, `--use_bump`, `--num_samples`) to customize runs.
