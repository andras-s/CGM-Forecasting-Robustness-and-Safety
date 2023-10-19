import os

import numpy as np
import pandas as pd


run_1_save_string = 'run_20220810-161053_LSTM_i6_hor0.5_b1024_lr0.001_g0.999_seed0_linearlinear'
run_2_save_string = 'run_20220810-161343_LSTM_i6_hor0.5_b1024_lr0.001_g0.999_seed0_linearlinear'

base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
mtype = 'regression'


results_1_path = os.path.join(base_dir, mtype, run_1_save_string, 'results.xlsx')
results_2_path = os.path.join(base_dir, mtype, run_2_save_string, 'results.xlsx')

results_1 = pd.read_excel(results_1_path)
results_2 = pd.read_excel(results_2_path)

### Error Grid Type Metrics (lists)
column = 'test_PEG [% in A-E]'
folds = ['mean', 'std']

rel_entries = {key: {} for key in folds}
for fold in folds:
    relevant_entries_1 = results_1.loc[results_1['fold'] == fold, column].item().strip('][').split()[:-1]
    relevant_entries_2 = results_2.loc[results_2['fold'] == fold, column].item().strip('][').split()[:-1]
    relevant_entries_1 = [float(entry) if float(entry) != 0 else np.nan for entry in relevant_entries_1]
    relevant_entries_2 = [float(entry) for entry in relevant_entries_2]
    rel_differences_pct = [100 * (entry_2 - entry_1)/entry_1 for entry_1, entry_2 in zip(relevant_entries_1, relevant_entries_2)]

    rel_entries[fold]['entries_1'] = [round(ent, 3) if not np.isnan(ent) else 0. for ent in relevant_entries_1]
    rel_entries[fold]['entries_2'] = [round(ent, 3) if not np.isnan(ent) else 0. for ent in relevant_entries_2]
    rel_entries[fold]['rel_diffs'] = [round(rel, 1) if not np.isnan(rel) else 0. for rel in rel_differences_pct]


relevant_entries_1_tex = [f'{ent_fold_1} ({ent_fold_2})' for ent_fold_1, ent_fold_2 in zip(rel_entries['mean']['entries_1'], rel_entries['std']['entries_1'])]
relevant_entries_2_tex = [f'{ent_fold_1} ({ent_fold_2})' for ent_fold_1, ent_fold_2 in zip(rel_entries['mean']['entries_2'], rel_entries['std']['entries_2'])]
rel_differences_pct_tex = [r"\textit{" + f'{ent_fold_1}' + "}" for ent_fold_1, ent_fold_2 in zip(rel_entries['mean']['rel_diffs'], rel_entries['std']['rel_diffs'])]

relevant_entries_1_tex = ' & '.join(relevant_entries_1_tex) + r" \\ "
relevant_entries_2_tex = ' & '.join(relevant_entries_2_tex) + r" \\ "
rel_differences_pct_tex = ' & '.join(rel_differences_pct_tex) + r" \\ "

print(relevant_entries_1_tex)
print(relevant_entries_2_tex)
print(rel_differences_pct_tex)


temp = 0
