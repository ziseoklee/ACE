import os
import pandas as pd
import sys

def process_directory(subdir_path):
    """
    Finds the experiment CSV in a subdirectory, runs aggregation,
    and saves summary_stats.csv.
    """
    source_csv_name = None
    
    # 1. Find the source experiment CSV file
    # We loop through all files and find the *first* CSV
    # that is NOT named 'summary_stats.csv'.
    try:
        for filename in os.listdir(subdir_path):
            if filename.endswith('.csv') and filename != 'summary_stats.csv':
                source_csv_name = filename
                break # Found our source file
    except OSError as e:
        print(f"  [ERROR] Could not read files in {subdir_path}: {e}")
        return False

    if not source_csv_name:
        print(f"  [Skipped] No source CSV found in: {os.path.basename(subdir_path)}")
        return False

    result_csv_path = os.path.join(subdir_path, source_csv_name)
    stats_csv_path = os.path.join(subdir_path, 'summary_stats.csv')

    try:
        # 2. Read the source CSV
        results_csv = pd.read_csv(result_csv_path)

        # 3. Run the exact aggregation logic you provided
        stats_by_method = results_csv.groupby(['method', 'A', 'B']).agg({
            'W1':         ['min', 'mean', 'std'],
            'W2':         ['min', 'mean', 'std'],
            'MMD_RBF':    ['min', 'mean', 'std'],
            'Total Var.': ['min', 'mean', 'std']
        }).reset_index()
        
        # 4. Save the new summary CSV
        stats_by_method.to_csv(stats_csv_path, index=False)
        print(f"  [Success] Created summary in: {os.path.basename(subdir_path)}")
        return True
        
    except pd.errors.EmptyDataError:
        print(f"  [ERROR] Source CSV is empty: {result_csv_path}")
        return False
    except KeyError as e:
        print(f"  [ERROR] Column not found (e.g., 'method' or 'W1') in {result_csv_path}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to process {result_csv_path}: {e}")
        return False

def main(parent_dir):
    """
    Goes through all subdirectories in the parent_dir and
    processes them.
    """
    if not os.path.isdir(parent_dir):
        print(f"Error: Path provided is not a valid directory: {parent_dir}")
        sys.exit(1)
    
    print(f"Processing all subdirectories inside: {parent_dir}\n")
    
    processed_count = 0
    skipped_count = 0

    # Loop through all items in the parent directory
    for item_name in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, item_name)
        
        # Check if the item is a directory
        if os.path.isdir(subdir_path):
            if process_directory(subdir_path):
                processed_count += 1
            else:
                skipped_count += 1
    
    print(f"\n--- Done ---")
    print(f"Successfully processed: {processed_count} directories.")
    print(f"Skipped/Failed:         {skipped_count} directories.")

if __name__ == "__main__":
    # Get the parent directory from the user
    parent_directory_path = input("Enter the path to the parent results directory (e.g., './'): ")
    
    # Clean up the path (e.g., remove quotes)
    parent_directory_path = parent_directory_path.strip().strip("'\"")
    
    main(parent_directory_path)