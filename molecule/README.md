# ACE for SBDD

- data: We use processed CrossDocked2020 data of Delete in https://zenodo.org/records/7980002

## Usage

### Install dependencies

```bash
bash scripts/setup_project.sh
```

### Run inference and postprocessing for CrossDocked2020 benchmark

Override configurations as needed. For example, to use ACE sampler with omega=1.4 for both sampler and weight, run:

```bash
python src/inference/run_inference.py weight.omega=1.4 sampler=ACESampler
```
