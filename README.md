# 2D-HCG
2D Heterogeneous Conditional Generation Examples

# Beginning a new session
```bash
conda update -n base -c conda-forge conda
conda env create -f environment.yml
conda activate 2D-HCG
```

# Ending a session
```bash
conda env export -n 2D-HCG --from-history > environment.yml
conda deactivate
conda env remove --name 2D-HCG
```

# Image HCG

```bash
python3 -m venv Image-HCG
source Image-HCG/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```