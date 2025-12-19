# tmux new -d './scripts/run_checker_eval.sh'
conda init
conda activate 2D-HCG
python eval_checker.py > eval_checker.log 2>&1