import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import sys
import re

# --- Configuration ---
# These can now be overridden by command-line args if you want,
# but we'll use them as defaults.
NUM_JOBS = 24      # Total number of jobs (len(collapse_combinations))
NUM_GPUS = 4       # Number of physical GPUs
# Calculate total parallel workers
MAX_WORKERS = (NUM_JOBS // NUM_GPUS) * NUM_GPUS if NUM_JOBS >= NUM_GPUS else NUM_JOBS
# This logic ensures you run e.g., 6 jobs/GPU if NUM_JOBS=24, NUM_GPUS=4
# Or if you set NUM_JOBS=12, MAX_WORKERS will be 12.
# Let's simplify: if you have 6 jobs/GPU, set MAX_WORKERS = 24
MAX_WORKERS = 24 # 6 jobs/GPU * 4 GPUs = 24 workers

MAX_WORKERS = NUM_GPUS
# ---------------------

def run_job(job_index, FAST_EVAL, RUN_METRIC_EVAL, BUMP_VALUE, ANNEAL_WEIGHT, ESS_THRESHOLD, NUM_STEPS, BATCH_SIZE):
    """
    Runs a single experiment job, assigning a GPU and creating a
    dedicated log file.
    """
    # Assigns GPU 0, 1, 2, 3, 0, 1, 2, 3, ...
    gpu_id = job_index % NUM_GPUS 
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # --- Arguments must be strings for subprocess ---
    command = [
        "python", 
        "251114_run_experiments.py", 
        "--job_index", str(job_index),
        # --- Fixed Boolean Flags ---
        # Only add the flag if it's True, which is what the worker's
        # 'action="store_true"' will expect.
    ]
    
    command.extend([
        "--FAST_EVAL", str(FAST_EVAL),
        "--RUN_METRIC_EVAL", str(RUN_METRIC_EVAL),
        "--BUMP_VALUE", str(BUMP_VALUE),
        "--ANNEAL_WEIGHT", str(ANNEAL_WEIGHT),
        "--ESS_THRESHOLD", str(ESS_THRESHOLD),
        "--NUM_STEPS", str(NUM_STEPS),
        "--BATCH_SIZE", str(BATCH_SIZE)
    ])

    # --- Improved Log Directory ---
    # Your original log_dir name was too long. Let's make a cleaner one.
    log_dir_name = f"logs_BUMP={BUMP_VALUE}_STEPS={NUM_STEPS}_BS={BATCH_SIZE}_ANNEAL_{ANNEAL_WEIGHT}_ESS={ESS_THRESHOLD}_FASTEVAL={FAST_EVAL}_METRICEVAL={RUN_METRIC_EVAL}"
    # Sanitize the name for file systems
    log_dir_name = re.sub(r"[^\w\.-]", "_", log_dir_name)
    
    if not os.path.exists(log_dir_name):
        try:
            os.makedirs(log_dir_name)
        except FileExistsError:
            pass # Race condition, another process made it
            
    log_filename = f"{log_dir_name}/job_{job_index:02d}.log" # e.g., job_00.log
    
    try:
        with open(log_filename, "w") as log_file:
            subprocess.run(
                command, 
                stdout=log_file, 
                stderr=subprocess.STDOUT, 
                check=True,  # Raise an error if the command fails
                env=env,
                text=True
            )
        return f"Job {job_index} (GPU {gpu_id}) SUCCESS"
        
    except subprocess.CalledProcessError as e:
        return f"Job {job_index} (GPU {gpu_id}) FAILED (see {log_filename})"
    except Exception as e:
        return f"Job {job_index} (GPU {gpu_id}) LAUNCH FAILED: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HCG experiments in parallel.")
    
    # --- FIXED Boolean Arguments ---
    # Default is True, so we add a flag to *disable* it (turn it to False)
    parser.add_argument("--no_fast_eval", action="store_true",
                        help="Disable fast evaluation (run full eval).")
    parser.add_argument("--no_metric_eval", action="store_true",
                        help="Disable metric evaluation.")

    # --- Other Arguments ---
    parser.add_argument("--BUMP_VALUE", type=float, default=20.0,
                        help="The Bump value to use in the experiment.")
    parser.add_argument("--ANNEAL_WEIGHT", type=float, default=1.0,
                        help="The Anneal weight to use in the experiment.")
    parser.add_argument("--ESS_THRESHOLD", type=float, default=0.7,
                        help="The ESS threshold to use in the experiment.")
    parser.add_argument("--NUM_STEPS", type=int, default=1000,
                        help="The number of steps to use in the experiment.")
    parser.add_argument("--BATCH_SIZE", type=int, default=10000,
                        help="The batch size to use in the experiment.")
    
    # --- Config for the launcher itself ---
    parser.add_argument("--num_jobs", type=int, default=NUM_JOBS,
                        help="Total number of jobs to run.")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help="Max number of parallel processes.")

    args = parser.parse_args()

    # --- Handle boolean logic ---
    # We want the *default* to be True, so the flag --no_fast_eval sets it to False.
    FAST_EVAL = not args.no_fast_eval
    RUN_METRIC_EVAL = not args.no_metric_eval

    print(f"Launching {args.num_jobs} jobs in parallel (max {args.max_workers} at a time)...")
    print(f"Settings: BUMP={args.BUMP_VALUE}, STEPS={args.NUM_STEPS}, FAST_EVAL={FAST_EVAL}")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        
        # --- THE KEY FIX ---
        # Pass all the parsed args to the 'run_job' function
        futures = {
            executor.submit(
                run_job, 
                i,                          # job_index
                FAST_EVAL,
                RUN_METRIC_EVAL,
                args.BUMP_VALUE,
                args.ANNEAL_WEIGHT,
                args.ESS_THRESHOLD,
                args.NUM_STEPS,
                args.BATCH_SIZE
            ) for i in range(args.num_jobs) # Use args.num_jobs
        }
        
        for future in tqdm(as_completed(futures), total=args.num_jobs):
            result = future.result()
            # Optional: print(result)
            pass 

    print("All jobs completed.")