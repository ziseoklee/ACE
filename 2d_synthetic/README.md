# 2D Synthetic Experiments

All commands assume `ace_env` has been activated (`source ace_env/bin/activate`) and the current working directory is `2d_synthetic`. If not, run:
```bash
cd 2d_synthetic
```

**Prerequisites (Toy Experiments)**
To run the pretraining code which will save checkpoints under `/PretrainedToyModels` (Takes ~30 mins on a single RTX A6000):

```bash
python ace_lib/train_toy_models.py
```

### Reproduce Results
To reproduce main evaluation results from the paper, run the following command:
```bash
python ace_eval_script.py
```

### Reproduce Figures
To reproduce main figures from the paper, please run each cell in the following python notebook in order:
```
ace_demo.ipynb
```