# ACE: Adaptive path Correction with Exponents
Ratio-of-densities steering for Diffusion/Flow matching models:
* 2D Synthetic Dataset
* Molecule optimization
* Compositional Image Generation

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Rebuttal Experiments
```bash
source venv/bin/activate

# Run this once to see the setup and get the total job count
# python 251114_run_experiments.py --job_index 0
python 251114_launcher.py \
    --BUMP_VALUE 20.0 \
    --ANNEAL_WEIGHT 2.0 \
    --ESS_THRESHOLD 0.7 \
    --num_jobs 24 \
    --no_fast_eval \ # This does not work!!!!
    > 251114_launcher_anneal2_full.log 2>&1 &

python 251114_launcher.py \
    --BUMP_VALUE 20.0 \
    --ANNEAL_WEIGHT 2.0 \
    --ESS_THRESHOLD 0.7 \
    --num_jobs 24 \
    > 251114_launcher_anneal2_fast.log 2>&1 &

# python 251114_launcher.py \
#     --BUMP_VALUE 20.0 \
#     --ANNEAL_WEIGHT 1.5 \
#     --ESS_THRESHOLD 0.7 \
#     --num_jobs 24 \
#     > 251114_launcher_anneal15.log 2>&1 &

python 251114_launcher.py \
    --BUMP_VALUE 20.0 \
    --ANNEAL_WEIGHT 3.0 \
    --ESS_THRESHOLD 0.7 \
    --num_jobs 24 \
    --no_fast_eval \
    > 251114_launcher_anneal3_full.log 2>&1 &

python 251114_launcher.py \
    --BUMP_VALUE 20.0 \
    --ANNEAL_WEIGHT 4.0 \
    --ESS_THRESHOLD 0.7 \
    --num_jobs 24 \
    > 251114_launcher_anneal4.log 2>&1 &

python 251114_launcher.py \
    --BUMP_VALUE 20.0 \
    --ANNEAL_WEIGHT 5.0 \
    --ESS_THRESHOLD 0.7 \
    --num_jobs 24 \
    > 251114_launcher_anneal5.log 2>&1 &

# This command will run jobs 0, 1, 2, ... 11, with 4 running at a time.
# Adjust -P4 to the number of parallel jobs you want (e.g., number of GPUs).
# seq 0 11 | xargs -n 1 -P 4 python run_experiment.py --job_index
```