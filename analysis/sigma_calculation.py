import os
import time
import pandas as pd
from analysis.helper import calc_fe_output, calc_sigma, calc_sigma_results


# Inputs
base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
mtype = 'regression'
model = 'LSTM'
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/LSTM_single'

seeds = [0, 1, 2, 3, 4]
horizons = [0.5, 1, 2]
folds = [0, 1, 2, 3]
gpu = '0'


# Calculate
start_time = time.strftime('%Y%m%d-%H%M%S')
sigmas = []

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

        for fold in folds:
            print(f'\n\n-----   Seed: {s}/{seeds}   Horizon: {h} in {horizons}   Fold: {fold} in {folds}   -----   \n\n')
            fe_ouput = calc_fe_output(run_path=run_path, base_dir=base_dir, fold=fold, gpu=gpu)
            fe_sigma = calc_sigma(fe_ouput=fe_ouput, model_name=model, fold=fold, horizon=h, model_seed=s)
            sigmas.append(fe_sigma.copy())


sigmas_df = pd.concat(sigmas)
sigmas_df.to_excel(os.path.join(base_dir, mtype, f"sigmas_{model}_{start_time}.xlsx"))
print(sigmas_df)
calc_sigma_results(os.path.join(base_dir, mtype, f"sigmas_{model}_{start_time}.xlsx"))

