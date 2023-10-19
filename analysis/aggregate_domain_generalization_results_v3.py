import os
import ast
import re

import numpy as np
import pandas as pd

new_gdu_models = True
gdu_models_dir = "LSTM_Layer_v2" if new_gdu_models else "LSTM_Layer"
experiment_paths = {
    "ERM_single  ": "/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/LSTM_single",
    "ERM_ensemble": "/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/LSTM_ensemble",
    "FT_CS       ": f"/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/{gdu_models_dir}/FT_CS",
    "FT_MMD      ": f"/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/{gdu_models_dir}/FT_MMD",
    "E2E_CS      ": f"/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/{gdu_models_dir}/E2E_CS",
    "E2E_MMD     ": f"/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/{gdu_models_dir}/E2E_MMD",
}

seeds = [0, 1, 2, 3, 4]
horizons = [0.5, 1, 2]
domains = ['BI', 'MDI', 'CSII', 'AP']
kpi_names = ['NLL', 'RMSE', 'pointwise NLL', 'pointwise RMSE']


def read_kpi_values(kpi, horizon, model_path, fold, domain, seed):
    seed_folders = os.listdir(model_path)
    seed_folder_name = [folder for folder in seed_folders if f'_{seed}_' in folder or f'seed_{seed}' in folder][0]
    seed_path = os.path.join(model_path, seed_folder_name)
    horizon_folders = os.listdir(seed_path)
    horizon_folder_name = [folder for folder in horizon_folders if f'_{seed}_{horizon}_' in folder or f'horizon_{horizon}' in folder][0]
    horizon_path = os.path.join(seed_path, horizon_folder_name)
    result_path = os.path.join(horizon_path, 'results.xlsx')
    results = pd.read_excel(result_path)
    value = results.loc[results['fold'] == fold, domain + '_' + kpi].iloc[0]
    if kpi in ['pointwise RMSE', 'pointwise NLL']:
        list_string = re.sub(r'\s+', ',', value)
        value = ast.literal_eval(list_string)[-1]
    return value


results = {}
for kpi in kpi_names:
    results[kpi] = {}
    print(f'\n-------------------------------------------------------')
    print(f'\n\n--------------------     {kpi}     --------------------')
    for horizon in horizons:
        results[kpi][horizon] = {}
        print(f'\n--------------------     {horizon}     --------------------')
        for model_name, model_path in experiment_paths.items():
            results[kpi][horizon][model_name] = {}
            for fold, domain in enumerate(domains):
                results[kpi][horizon][model_name][domain] = {}
                for seed in seeds:
                    value = read_kpi_values(kpi, horizon, model_path, fold, domain, seed)
                    results[kpi][horizon][model_name][domain][seed] = value
                seed_values = list(results[kpi][horizon][model_name][domain].values())
                results[kpi][horizon][model_name][domain]['avg'] = np.mean(seed_values)
                results[kpi][horizon][model_name][domain]['std'] = np.std(seed_values)

            if "ERM" in model_name:
                model_value_strings = ["\\textit{" + f"{round(results[kpi][horizon][model_name][domain]['avg'], 4):.4f} ({round(results[kpi][horizon][model_name][domain]['std'], 4):.4f})" + "}" for domain in domains]
            else:
                model_value_strings = [f"{round(results[kpi][horizon][model_name][domain]['avg'], 4):.4f} ({round(results[kpi][horizon][model_name][domain]['std'], 4):.4f})" for domain in domains]
            model_values_string = model_name + "   " + ' & '.join(model_value_strings) + r" \\ "
            print(model_values_string)
