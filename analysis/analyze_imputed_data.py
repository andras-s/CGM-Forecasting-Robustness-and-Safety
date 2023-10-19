import os
import pandas as pd
import torch

from utils.setup import DATA_DIR
from utils.calc import MMOLL_MGDL, glucose_levels


imputed_data_dir_name = '20220812-112318'
save_dir = 'results/data_kpi_analysis/'


imputed_data_dir_path = os.path.join(DATA_DIR, '4_imputed_data', imputed_data_dir_name)
train_data = pd.read_pickle(os.path.join(imputed_data_dir_path, f'train_data_fold_{0}.pkl'))
val_data = pd.read_pickle(os.path.join(imputed_data_dir_path, f'val_data_fold_{0}.pkl'))
test_data = pd.read_pickle(os.path.join(imputed_data_dir_path, f'test_data_fold_{0}.pkl'))

data_dfs = []
for split_name, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
    info_df = data.window_data[['window_id', 'file_id', 'treatment', 'diabetes_type', 'diabetes_duration', 'sex', 'age']]

    data.inputs = data.inputs.squeeze()
    data.target = data.target.squeeze()[:, 1:]
    entries = torch.cat([data.inputs, data.target], dim=1)
    column_names = [f'glucose_value_{i}' for i in range(entries.size(1))]
    data_df = pd.DataFrame(entries, columns=column_names)

    split = pd.DataFrame([split_name] * entries.size(0), columns=['split'])

    df = pd.concat([split, info_df, data_df], axis=1)
    data_dfs.append(df)

data_df = pd.concat(data_dfs, axis=0).reset_index(drop=True)

domain_names = []
for ind in range(len(data_df)):
    treatment = data_df.loc[ind, 'treatment']
    if treatment in ['basal', 'basal only', 'insulin+nia']:
        domain = 'BI'
    elif treatment in ['MDI']:
        domain = 'MDI'
    elif treatment in ['CSII', 'Medtronic 640G']:
        domain = 'CSII'
    elif treatment in ['HYBRID_mm670g', 'HYBRID_mm780g', 'HYBRID_accucheck']:
        domain = 'AP'
    else:
        domain = 'unknown'
    domain_names.append(domain)

data_df['domain'] = domain_names
domains = ['BI', 'MDI', 'CSII', 'AP']
eval_domains = ['all'] + domains
print(data_df['domain'].value_counts())


num_windows = {}
avg_glucose = {}
std_glucose = {}
pct_hypo = {}
pct_hyper = {}

hypo_thr = glucose_levels['hypo L1'][1] / MMOLL_MGDL
hyper_thr = glucose_levels['hyper L1'][0] / MMOLL_MGDL

for domain in eval_domains:
    domain_str = domain if domain != 'all' else ''

    df = data_df.loc[data_df.domain.str.contains(domain_str)]
    entries = df.loc[:, 'glucose_value_0':].iloc[:, :-1].to_numpy()

    num_windows[domain] = int(len(df))
    avg_glucose[domain] = round(entries.mean(), 2)
    std_glucose[domain] = round(entries.std(), 2)
    pct_hypo[domain] = round(100 * (entries <= hypo_thr).sum() / entries.size, 2)
    pct_hyper[domain] = round(100 * (entries >= hyper_thr).sum() / entries.size, 2)

print(num_windows, avg_glucose, std_glucose, pct_hypo, pct_hyper)

results = pd.DataFrame([num_windows, avg_glucose, std_glucose, pct_hypo, pct_hyper], index=['Days', 'Avg. CGM', 'Std. CGM', '% Hypo', '% Hyper'])
# results.to_excel(save_dir + 'kpis.xlsx')
