# ACE compositional image generation

This directory reproduces the Stable Diffusion rows of Table E.10 in the ACE paper, *On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering*. It compares the same compositional sampler under three settings:

- **NR**: composed score propagation without importance weighting or resampling.
- **FKC**: constant exponents with Feynman–Kac importance weighting.
- **ACE**: FKC with the endpoint-preserving exponent correction
  `gamma_i(t) + 5 t (1-t)`.

`HFKC` was an early development name. Old imports and the `gen-mig-box-hfkc` command remain as compatibility aliases, but new commands use `ACE` and `FKC`.

## Paper settings

The reproduction configs encode the settings reported in Appendix E.4:

| Setting | Value |
|---|---:|
| Backbones | Stable Diffusion 1.5 and 2.1 |
| Denoising steps | 50 |
| Global guidance | 7.5 |
| Local guidance | 7.5 |
| Particles | 3 |
| DDIM stochasticity (`eta`) | 1.5 |
| ACE bump (`B`) | 5 |
| Resampling | once at generation progress `t=0.3` |
| COCO-MIG seeds | 42, 37, 519, 609, 123, 401, 780, 0 |

The implementation keeps the backbone, schedule, prompts, particles, and random seeds identical across NR, FKC, and ACE. FKC is exactly the `B=0` special case; ACE additionally tracks the local conditional/unconditional log-density ratios required by the `gamma_dot_i(t) log q_i(t)` term in Algorithm 1. Scheduled resampling preserves particle ancestry for both latents and tracked log densities.

## Installation

Python 3.11, Git, and an NVIDIA CUDA GPU are required. The lockfile selects the CUDA 12.4 PyTorch wheels. Model and evaluator downloads require network access and several gigabytes of free disk space. From this directory:

```bash
uv venv --python 3.11
uv sync --frozen --cache-dir .cache
```

Stable Diffusion 2.1 may require accepting its Hugging Face model license and authenticating first:

```bash
uv run hf auth login
```

The COCO-MIG JSONL is included at `data/mig_bench.jsonl`. Before evaluation, download the GroundingDINO checkpoint expected by the bundled evaluator:

```bash
uv run bash scripts/download_evaluation_weights.sh
```

Stable Diffusion, CLIP, and SAM weights are downloaded on first use.

## Quick smoke run

This runs one COCO-MIG difficulty level and one seed. It is the recommended check before launching the full benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 uv run gen-mig-box-ace \
  method=sd21+ace \
  'seed=[42]' \
  'target_levels=[0]' \
  run_name=smoke_sd21_ace
```

Generation is resumable: rerunning the same command keeps the deterministic output directory and skips matching images already present.

## Reproduce Table E.10

Run all six paper configurations:

```bash
CUDA_VISIBLE_DEVICES=0 uv run bash scripts/reproduce_coco_mig.sh
```

Outputs are written to fixed directories:

```text
output/COCO-MIG_Bench/
├── sd15_nr
├── sd15_fkc
├── sd15_ace
├── sd21_nr
├── sd21_fkc
└── sd21_ace
```

Run a subset by overriding the script variables. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
ACE_BACKBONES='sd21' \
ACE_SAMPLERS='fkc ace' \
ACE_LEVELS='0,1' \
uv run bash scripts/reproduce_coco_mig.sh
```

Each complete configuration is computationally expensive. Use separate output roots if distributing configurations across machines. Multiple processes must not write the same `run_name` concurrently.

## Evaluate generated images

Evaluate all six directories with the same eight iterations used for generation:

```bash
uv run bash scripts/download_evaluation_weights.sh
CUDA_VISIBLE_DEVICES=0 uv run bash scripts/evaluate_coco_mig.sh
```

Metric JSON files are saved under `results/COCO-MIG_Bench/`. To evaluate only SD2.1 ACE:

```bash
CUDA_VISIBLE_DEVICES=0 \
ACE_BACKBONES='sd21' \
ACE_SAMPLERS='ace' \
uv run bash scripts/evaluate_coco_mig.sh
```

The paper reports the following averages as regression targets. Values are instance attribute success ratio, with mIoU in parentheses:

| Method | SD1.5 | SD2.1 |
|---|---:|---:|
| NR | 20.24 (26.53) | 21.04 (26.93) |
| FKC | 28.27 (31.51) | 31.99 (33.95) |
| ACE (`B=5`) | 37.84 (36.67) | 40.46 (38.33) |

Small numerical differences can arise from GPU kernels and dependency builds. Large differences usually indicate a changed model revision, seed list, benchmark data, or generation parameters; each run writes its resolved configuration to a log beside the output directory.

## Individual demo

The standalone CLI is useful for inspecting a custom layout:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baselines/demo_sd_ace.py \
  --arch SD2.1 \
  --sampler ace \
  --bump 5 \
  --N 3 \
  --eta 1.5 \
  --resample-at 0.3 \
  --prompt 'A photo of a cow, a horse, and a sheep on a green field' \
  --tags cow horse sheep \
  --phrases 'A photo of a cow' 'A photo of a horse' 'A photo of a sheep' \
  --boxes 0.05 0.55 0.28 0.90 0.36 0.52 0.64 0.90 0.72 0.54 0.95 0.90 \
  --output-dir figures/cow_horse_sheep
```

Change only `--sampler` to `fkc` or `nr` for a controlled comparison. `--bump` is ignored outside ACE.

## Code map

- `src/baselines/demo_sd_ace.py`: shared NR/FKC/ACE particle sampler.
- `src/ace_schedule.py`: dependency-free bump and resampling schedule definitions.
- `src/configs/config_method.py`: the six paper configurations.
- `src/experiments/bench_coco_mig/generate_mig_textbox2img_ace.py`: deterministic COCO-MIG generation driver.
- `run_migbench_ace.py`: particle-aware COCO-MIG evaluator entry point.
- `tests/test_ace_schedule.py`: endpoint, derivative, and paper resampling-time checks.

## Validation

The schedule tests do not require downloading models:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests run_migbench_ace.py
```

Full numerical validation requires the CUDA environment and downloaded model weights because the sampler estimates Algorithm 1 divergence terms through the Stable Diffusion UNet.
