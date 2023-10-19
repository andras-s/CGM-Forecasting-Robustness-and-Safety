import os
import time
import pandas as pd
from analysis.helper import calc_fe_output, calc_clustering_results, plot_clustering_results

# Inputs
base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
mtype = 'regression'
model = 'LSTM'
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/LSTM_single'

seeds = [4]
horizons = [2]
folds = [0, 1, 2, 3]
min_clusters = 2
max_clusters = 50
gpu = '1'


# Calculate
start_time = time.strftime('%Y%m%d-%H%M%S')
clustering_results = []

seed_folders = os.listdir(experiment_path)
for s in seeds:
    seed_folder = [f_name for f_name in seed_folders if f'_{s}_' in f_name][0]
    seed_path = os.path.join(experiment_path, seed_folder)
    horizon_folders = os.listdir(seed_path)

    for h in horizons:
        horizon_folder = [f_name for f_name in horizon_folders if f'_{s}_{h}_' in f_name][0]
        run_path = os.path.join(seed_path, horizon_folder)
        results_path = os.path.join(run_path, 'evaluations')
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        run_clustering_results = []
        for fold in folds:
            print(f'\n\n-----   Seed: {s}/{seeds}   Horizon: {h} in {horizons}   Fold: {fold} in {folds}   -----   \n\n')
            fe_ouput = calc_fe_output(run_path=run_path, base_dir=base_dir, fold=fold, gpu=gpu)
            fe_clustering_result = calc_clustering_results(fe_ouput=fe_ouput, min_clusters=min_clusters, max_clusters=max_clusters, model_name=model, fold=fold, horizon=h, model_seed=s)
            run_clustering_results.append(fe_clustering_result.copy())

        run_clustering_results_df = pd.concat(run_clustering_results)
        run_clustering_results_df.to_excel(os.path.join(results_path, f'clustering_results_{start_time}.xlsx'))
        clustering_results.append(run_clustering_results_df.copy())

clustering_results_df = pd.concat(clustering_results)
clustering_results_df.to_excel(os.path.join(base_dir, mtype, f"clustering_results_{model}_{start_time}.xlsx"))
print(clustering_results_df)
plot_clustering_results(os.path.join(base_dir, mtype, f"clustering_results_{model}_{start_time}.xlsx"))
