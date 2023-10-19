import os
import ast
import re

import numpy as np
import pandas as pd

experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment'

models = ['t_0', 'ConvT', 'LSTM']                                   # ['t_0', 'ConvT', 'LSTM']
loss_names = ['NLL', 'NLLPEGSurface']
horizons = [0.5, 1, 2]                                              # [0.5, 1, 2]
kpi_names = ['RMSE', 'pointwise RMSE', 'NLL', 'pointwise NLL']
eg_kpi_names = ['PEG [% in A-E]', 'pointwise PEG [% in A-E]']

print(f"\nAggregating Results from models in : {experiment_path.split('/')[-1]}\n")
all_run_dirs = os.listdir(experiment_path)
for kpi in kpi_names:
    print(f'\n--------------------     {kpi}     --------------------')

    for model in models:
        model_run_dirs = [dir for dir in all_run_dirs if model in dir]

        losses = ['t_0'] if model == 't_0' else loss_names
        for loss in losses:
            loss_run_dirs = [dir for dir in model_run_dirs if loss + '_' in dir]

            kpis = {
                horizon: {
                    'mean': [],
                    'std': []
                }
                for horizon in horizons
            }
            for horizon in horizons:
                run_dir = [dir for dir in loss_run_dirs if 'hor' + str(horizon) in dir][-1]
                result_path = os.path.join(experiment_path, run_dir, 'results.xlsx')
                run_results = pd.read_excel(result_path)
                mean_value = run_results.loc[run_results['fold'] == 'mean', 'test_' + kpi].iloc[0]
                std_value = run_results.loc[run_results['fold'] == 'std', 'test_' + kpi].iloc[0]
                if kpi in ['pointwise RMSE', 'pointwise NLL']:
                    mean_list_string = re.sub(r'\s+', ',', mean_value)
                    std_list_string = re.sub(r'\s+', ',', std_value)
                    mean_value = ast.literal_eval(mean_list_string)[-1]
                    std_value = ast.literal_eval(std_list_string)[-1]
                kpis[horizon]['mean'] = mean_value
                kpis[horizon]['std'] = std_value

            print_kpis = {horizon: f"{round(kpis[horizon]['mean'], 3):.3f} ({round(kpis[horizon]['std'], 3):.3f})" for horizon in horizons}
            latex_format = ' & '.join(list(print_kpis.values())) + r" \\ "
            # print(f'{model} {loss} ', print_kpis)
            print(f'{model}   {loss} ', latex_format)


def get_values(results_df, fold, train_val_test, kpi):
    if 'pointwise' not in kpi:
        values = results_df.loc[results_df.fold == fold, train_val_test + kpi].item().strip('][').split()[:-1]
    else:
        values = results_df.loc[results_df.fold == fold, train_val_test + kpi].item().strip('][').split('\n ')[-1].strip('][').split()[:-1]
    values = [float(entry) if float(entry) != 0 else np.nan for entry in values]
    return values


for kpi in eg_kpi_names:
    print(f'\n--------------------     {kpi}     --------------------')

    for horizon in horizons:
        print(f'\n              ----     {horizon}     ---              ')

        for model in models:

            if model == 't_0':
                run_dir = [dir for dir in all_run_dirs if 'hor' + str(horizon) in dir and model in dir][0]
                result_path = os.path.join(experiment_path, run_dir, 'results.xlsx')
                results = pd.read_excel(result_path)
                means = get_values(results, fold='mean', train_val_test='test_', kpi=kpi)
                means = [round(ent, 3) if not np.isnan(ent) else 0. for ent in means]
                stds = get_values(results, fold='std', train_val_test='test_', kpi=kpi)
                stds = [round(ent, 3) if not np.isnan(ent) else 0. for ent in stds]
                output_latex = [f'{mean:.3f} ({std:.3f})' for mean, std in zip(means, stds)]
                output_latex = ' & '.join(output_latex) + r" \\ "
                print(f'{model}     ', output_latex)

            elif model in ['ConvT', 'LSTM']:
                nll_dir = [dir for dir in all_run_dirs if 'hor' + str(horizon) in dir and model in dir and '_NLL_' in dir][0]
                nll_result_path = os.path.join(experiment_path, nll_dir, 'results.xlsx')
                nll_results = pd.read_excel(nll_result_path)
                nll_mean_values = get_values(nll_results, fold='mean', train_val_test='test_', kpi=kpi)
                nll_std_values = get_values(nll_results, fold='std', train_val_test='test_', kpi=kpi)

                peg_dir = [dir for dir in all_run_dirs if 'hor' + str(horizon) in dir and model in dir and '_NLLPEGSurface_' in dir][0]
                peg_result_path = os.path.join(experiment_path, peg_dir, 'results.xlsx')
                peg_results = pd.read_excel(peg_result_path)
                peg_mean_values = get_values(peg_results, fold='mean', train_val_test='test_', kpi=kpi)
                peg_std_values = get_values(peg_results, fold='std', train_val_test='test_', kpi=kpi)

                rel_diff_means = [100 * (entry_2 - entry_1) / entry_1 for entry_1, entry_2 in zip(nll_mean_values, peg_mean_values)]
                # diff_std = [100 * (entry_2 - entry_1) / entry_1 for entry_1, entry_2 in zip(nll_std_values, peg_std_values)]

                nll_mean_values = [round(ent, 3) if not np.isnan(ent) else 0. for ent in nll_mean_values]
                nll_std_values = [round(ent, 3) if not np.isnan(ent) else 0. for ent in nll_std_values]
                peg_mean_values = [round(ent, 3) if not np.isnan(ent) else 0. for ent in peg_mean_values]
                peg_std_values = [round(ent, 3) if not np.isnan(ent) else 0. for ent in peg_std_values]
                rel_diff_means = [round(rel, 1) if not np.isnan(rel) else 0. for rel in rel_diff_means]
                # diff_std = [round(rel, 1) if not np.isnan(rel) else 0. for rel in diff_std_pct]

                nll_output_latex = [f'{mean:.3f} ({std:.3f})' for mean, std in zip(nll_mean_values, nll_std_values)]
                peg_output_latex = [f'{mean:.3f} ({std:.3f})' for mean, std in zip(peg_mean_values, peg_std_values)]
                rel_diff_latex = [r"\textit{" + f'{reL_diff:.1f}' + "}" for reL_diff in rel_diff_means]
                nll_output_latex = ' & '.join(nll_output_latex) + r" \\ "
                peg_output_latex = ' & '.join(peg_output_latex) + r" \\ "
                rel_diff_latex = ' & '.join(rel_diff_latex) + r" \\ "

                print(f'{model} NLL ', nll_output_latex)
                print(f'{model} PEG ', peg_output_latex)
                print(f'{model} rel ', rel_diff_latex, '\n')
