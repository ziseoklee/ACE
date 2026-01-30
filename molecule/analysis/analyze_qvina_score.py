import torch
import pandas as pd
import argparse
import numpy as np
import sys
import os
import ast
from utils.utils import sandbox_import_dir
from utils.utils import nested_to_df, get_project_root

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    args = parser.parse_args(argv)
    task_list = []
    df = pd.read_csv(args.csv_path)
    print(f'{args.csv_path} has {df.shape[0]} rows')
    best_scores = []
    top3_scores =  []
    worst_scores = []
    mean_scores = []
    median_scores = []
    std_scores = []
    all_scores = []
    optimized_scores = []
    
    nonzero_best_scores = []
    nonzero_top3_scores = []
    nonzero_optimized_scores = []
    nonzero_worst_scores = []
    nonzero_mean_scores = []
    nonzero_median_scores = []
    nonzero_std_scores = []
    nonzero_all_scores = []

    project_root = get_project_root()
    data_root = os.path.join(project_root, 'data', 'crossdock', 'raw')
    protein_dir = os.path.join(project_root, 'src', 'pretrained_models', 'DiffSBDD', 'dataset', 'crossdock', 'crossdocked_pocket10')
    analysis_res_dir = os.path.join(project_root, 'result_analysis')
    analyze_df = pd.read_csv(os.path.join(analysis_res_dir, 'crossdock', 'len_dict.csv'))
    fail_cnt = 0
    complete_fail_cnt = 0
    task_list = [14, 15, 27, 38, 43, 60, 69, 77, 78]
    final_task_list = []
    for i in task_list:
        data_path = os.path.join(data_root, f'{i}.pt')
        if not os.path.exists(data_path):
            continue
        affinity = ast.literal_eval(analyze_df.loc[analyze_df['index'] == i]['length'].iloc[0])['affinity']
        if affinity < -5:
            continue
        with sandbox_import_dir(os.path.join(project_root, 'baseline', 'Delete')):
            data = torch.load(data_path)
        scaffold_length = data.scaffold.GetNumAtoms()
        if scaffold_length < 7:
            continue
        scores = df.loc[df.iloc[:, 0] == i]['scores'].iloc[0]
        scores = np.fromstring(scores.strip("[]"), sep=" ")

        final_task_list.append(i)


        if len(scores) == 0:
            complete_fail_cnt += 1
            fail_cnt += 5
            continue

        best_scores.append(scores[0])
        top3_scores.append(scores[:3].mean())
        optimized_scores.append((scores <= affinity).sum())
        worst_scores.append(scores[-1])
        optimized_scores.append((scores <= affinity).sum())
        mean_scores.append(np.mean(scores))
        median_scores.append(np.median(scores))
        std_scores.append(np.std(scores))
        all_scores.append(scores)

        for score in scores:
            if score > -0.01:
                fail_cnt += 1

        if scores[0] > -0.01:
            complete_fail_cnt += 1
            continue
            
        nozero_scores = scores[scores != 0]
        nonzero_best_scores.append(nozero_scores[0])
        nonzero_top3_scores.append(nozero_scores[:3].mean())
        nonzero_optimized_scores.append((nozero_scores <= affinity).sum())
        nonzero_worst_scores.append(nozero_scores[-1])
        nonzero_mean_scores.append(np.mean(nozero_scores))
        nonzero_median_scores.append(np.median(nozero_scores))
        nonzero_std_scores.append(np.std(nozero_scores))
        nonzero_all_scores.append(nozero_scores)

    print(f'best_scores: {np.mean(best_scores)}, {np.median(best_scores)}, {np.std(best_scores)}')
    print(f'top3_scores: {np.mean(top3_scores)}, {np.median(top3_scores)}, {np.std(top3_scores)}')
    print(f'optimized_scores: {np.mean(optimized_scores)}, {np.median(optimized_scores)}, {np.std(optimized_scores)}')
    print(f'worst_scores: {np.mean(worst_scores)}, {np.median(worst_scores)}, {np.std(worst_scores)}')
    print(f'mean_scores: {np.mean(mean_scores)}, {np.median(mean_scores)}, {np.std(mean_scores)}')
    print(f'median_scores: {np.mean(median_scores)}, {np.median(median_scores)}, {np.std(median_scores)}')
    print(f'std_scores: {np.mean(std_scores)}, {np.median(std_scores)}, {np.std(std_scores)}')

    # print(f'all_scores: {np.mean(all_scores)}, {np.median(all_scores)}, {np.std(all_scores)}')
    print(f'nonzero_best_scores: {np.mean(nonzero_best_scores)}, {np.median(nonzero_best_scores)}, {np.std(nonzero_best_scores)}')
    print(f'nonzero_top3_scores: {np.mean(nonzero_top3_scores)}, {np.median(nonzero_top3_scores)}, {np.std(nonzero_top3_scores)}')
    print(f'nonzero_optimized_scores: {np.mean(nonzero_optimized_scores)}, {np.median(nonzero_optimized_scores)}, {np.std(nonzero_optimized_scores)}')
    print(f'nonzero_worst_scores: {np.mean(nonzero_worst_scores)}, {np.median(nonzero_worst_scores)}, {np.std(nonzero_worst_scores)}')
    print(f'nonzero_mean_scores: {np.mean(nonzero_mean_scores)}, {np.median(nonzero_mean_scores)}, {np.std(nonzero_mean_scores)}')
    print(f'nonzero_median_scores: {np.mean(nonzero_median_scores)}, {np.median(nonzero_median_scores)}, {np.std(nonzero_median_scores)}')
    print(f'nonzero_std_scores: {np.mean(nonzero_std_scores)}, {np.median(nonzero_std_scores)}, {np.std(nonzero_std_scores)}')
    # print(f'nonzero_all_scores: {np.mean(nonzero_all_scores)}, {np.median(nonzero_all_scores)}, {np.std(nonzero_all_scores)}')

    # save as csv
    df = pd.DataFrame([
        np.sum(complete_fail_cnt) / len(task_list),
        np.sum(fail_cnt) / ( 5 * len(task_list)),
        np.mean(best_scores),
        np.mean(top3_scores),
        np.mean(optimized_scores),
        np.mean(worst_scores),
        np.mean(mean_scores),
        np.mean(median_scores),
        np.mean(std_scores),
    ], index= ['complete_fail_rate', 'fail_rate', 'best_scores', 'top3_scores', 'optimized_scores', 'worst_scores', 'mean_scores', 'median_scores', 'std_scores']).T
    save_path = os.path.join(analysis_res_dir, os.path.basename(args.csv_path.replace('.csv', '_small_analysis.csv')))
    df.to_csv(save_path, index=False)

    df_nonzero = pd.DataFrame([   
        np.mean(nonzero_best_scores),
        np.mean(nonzero_top3_scores),
        np.mean(nonzero_optimized_scores),
        np.mean(nonzero_worst_scores),
        np.mean(nonzero_mean_scores),
        np.mean(nonzero_median_scores),
        np.mean(nonzero_std_scores),
    ], index= ['nonzero_best_scores', 'nonzero_top3_scores', 'nonzero_optimized_scores', 'nonzero_worst_scores', 'nonzero_mean_scores', 'nonzero_median_scores', 'nonzero_std_scores']).T
    save_path = os.path.join(analysis_res_dir, os.path.basename(args.csv_path.replace('.csv', '_small_analysis_nonzero.csv')))
    df_nonzero.to_csv(save_path, index=False)

    df = pd.DataFrame(final_task_list, index=final_task_list)
    save_path = os.path.join(analysis_res_dir, os.path.basename(args.csv_path.replace('.csv', '_task_list.csv')))
    df.to_csv(save_path, index=False)
    print(f'save_path: {save_path}')
if __name__ == '__main__':
    main(sys.argv[1:])
