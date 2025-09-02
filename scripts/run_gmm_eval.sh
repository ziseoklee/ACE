# tmux new -d './scripts/run_gmm_eval.sh'
conda init
conda activate 2D-HCG
python eval_gmm.py > eval_gmm.log 2>&1