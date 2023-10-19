import itertools
import os
import ast
import re
import statistics
import sys
import json
import pickle
import random
import time
import datetime

import numpy as np
import pandas as pd
from math import floor
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch import nn
import torch

from forecast.data import load_data
from forecast.helper import get_metrics, get_model
from forecast.evaluation import ModelEvaluator
from utils.setup import init_gpu, DOMAINS
from utils.utils import dotdict


def calc_clustering_results(fe_ouput, min_clusters, max_clusters, model_name, fold, horizon, model_seed):
    entries = {
        'model': [],
        'horizon': [],
        'fold': [],
        'model_seed': [],
        'num_clusters': [],
        'calinski_harabasz_score': [],
        'davies_bouldin_score': [],
        'silhouette_score': [],
    }

    K = range(min_clusters, max_clusters + 1)
    for num_clusters in K:
        kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++').fit(fe_ouput)
        labels = kmeans_model.labels_

        calinski_harabasz_score = metrics.calinski_harabasz_score(fe_ouput, labels)
        davies_bouldin_score = metrics.davies_bouldin_score(fe_ouput, labels)
        silhouette_score = metrics.silhouette_score(fe_ouput, labels, metric='euclidean')

        entries['model'].append(model_name)
        entries['horizon'].append(horizon)
        entries['fold'].append(fold)
        entries['model_seed'].append(model_seed)
        entries['num_clusters'].append(num_clusters)
        entries['calinski_harabasz_score'].append(calinski_harabasz_score)
        entries['davies_bouldin_score'].append(davies_bouldin_score)
        entries['silhouette_score'].append(silhouette_score)

        print(
            f'Clusters: {num_clusters}/{max_clusters}   Calinski-Harabasz: {calinski_harabasz_score}   Davies-Bouldin: {davies_bouldin_score}   Silhouette Score: {silhouette_score}')

    results = pd.DataFrame(entries)
    return results


def calc_sigma(fe_ouput, model_name, fold, horizon, model_seed):
    sigma = np.median(metrics.euclidean_distances(fe_ouput))
    entry = {
        'model': [model_name],
        'horizon': [horizon],
        'fold': [fold],
        'model_seed': [model_seed],
        'sigma': [sigma],
    }
    print(f'sigma: {sigma}')
    results = pd.DataFrame(entry)
    return results


def calc_fe_output(run_path, fold, gpu, data='train'):
    # Initialize
    with open(run_path + '/args.json') as file:
        args = json.load(file)
    args = dotdict(args)
    # args.base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
    if args.Layer_hps['fine_tuning_FEs_path']:
        args.Layer_hps['fine_tuning_FEs_path'] = os.path.join(
            # '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment',
            '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression',
            args.Layer_hps['fine_tuning_FEs_path'])
    sys.path.append(os.path.expanduser(args.repo_location))
    torch.backends.cudnn.enabled = False
    torch.manual_seed(0)
    # random.seed(args.seed)
    device = init_gpu(gpu)

    args.data_path = '/local/home/ansass/Thesis/icarus/analysis/data_on_server/v_2/4_imputed_data/' + args.data_path.split('/', -1)[-1] # todo: only works for ansass
    data_path = args.data_path

    # Get Data and Model
    train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        fold, data_path, args)
    model, optimiser, scheduler = get_model(args, device, train_data, train_dataset,
                                            num_input_features=train_data[0][0].size(1),
                                            input_length=train_data[0][0].size(0),
                                            output_size=train_data[0][-1].size(0), fold=fold)
    model.load_state_dict(torch.load(os.path.join(run_path, "models", f"weights_{fold}.pth")))
    feature_extractor = model.feature_extractor
    feature_extractor.eval()

    # Get FE output
    if data == 'train':
        inputs, target = train_data[:]
        inputs = inputs.float().to(device)
        with torch.no_grad():
            x = feature_extractor(inputs).cpu()
            x = x.cpu().detach().numpy()
        del inputs, target

    elif data == 'all_folds':
        in_out = calculate_model_outputs(model=feature_extractor, data=[train_data, val_data, test_data],
                                         model_num=fold)
        x = in_out

    del args, device, train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, model, optimiser, scheduler, feature_extractor
    torch.cuda.empty_cache()

    return x


def plot_clustering_results(path=None):
    if path is None:
        path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/clustering_results_LSTM.xlsx'
    df = pd.read_excel(path)

    best_num_clusters = []
    for h in df.horizon.unique():
        for f in df.fold.unique():
            for s in df.model_seed.unique():
                df_hfs = df.loc[(df.horizon == h) & (df.fold == f) & (df.model_seed == s)]
                max_score = df_hfs['calinski_harabasz_score'].max()
                best_num_clusters.append(
                    df_hfs.loc[df_hfs['calinski_harabasz_score'] == max_score, 'num_clusters'].item())
                print(f'horizon: {h} fold: {f} seed: {s} best number of clusters: {best_num_clusters[-1]}')

    print('Average the Calinsky-Harabasz Scores and take the Maximum of the average curve')
    for h in df.horizon.unique():
        for f in df.fold.unique():
            df_hf = df.loc[(df.horizon == h) & (df.fold == f)]
            seeds = df_hf.model_seed.unique()
            avg_calinsky_harabasz_scores = np.zeros(len(df_hf[df_hf.model_seed == seeds[0]]))
            for m_seed in seeds:
                avg_calinsky_harabasz_scores = avg_calinsky_harabasz_scores + df_hf.loc[
                    df_hf.model_seed == m_seed, 'calinski_harabasz_score'].to_numpy()
            avg_calinsky_harabasz_scores = 1 / len(seeds) * avg_calinsky_harabasz_scores
            avg_max_calinsky_harabasz_score = avg_calinsky_harabasz_scores.max()
            best_avg_clusters_position = avg_calinsky_harabasz_scores.argmax()
            best_avg_clusters = df_hf.num_clusters.iloc[best_avg_clusters_position]
            print(f'horizon: {h} fold: {f} best number of clusters: {best_avg_clusters}')
            axes = plt.gca()
            df_hf.groupby('model_seed').plot(kind='line', x='num_clusters', y='calinski_harabasz_score', ax=axes,
                                             title=f'horizon {h} fold {f}: AVG Score {round(avg_max_calinsky_harabasz_score)} at {best_avg_clusters} clusters')
            plt.show()

    print(
        'Take the Maximums of the individual models Calinsky-Harabasz Scores curve and then take the median from the 5 seeds:')
    for h in df.horizon.unique():
        for f in df.fold.unique():
            df_hf = df.loc[(df.horizon == h) & (df.fold == f)]
            seeds = df_hf.model_seed.unique()

            best_nums_clusters = []
            for m_seed in seeds:
                df_hfs = df.loc[(df.horizon == h) & (df.fold == f) & (df.model_seed == m_seed)]
                scores = df_hfs['calinski_harabasz_score'].to_numpy()
                best_num_clusters_pos = scores.argmax()
                best_nums_clusters.append(df_hfs.num_clusters.iloc[best_num_clusters_pos].item())

            best_median_num_clusters = statistics.median(best_nums_clusters)
            print(
                f'horizon: {h} fold: {f} best median number of clusters: {best_median_num_clusters} {best_nums_clusters}')
            axes = plt.gca()
            df_hf.groupby('model_seed').plot(kind='line', x='num_clusters', y='calinski_harabasz_score',
                                             ax=axes,
                                             title=f'horizon {h} fold {f}: median {best_median_num_clusters} clusters')
            plt.show()


def calc_sigma_results(path):
    if path is None:
        path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/sigmas_LSTM_20220814-184942.xlsx'
    df = pd.read_excel(path)

    entries = {
        'horizon': [],
        'fold': [],
        'sigma': [],
        'std': [],
    }
    for h in df.horizon.unique():
        for f in df.fold.unique():
            df_hf = df.loc[(df.horizon == h) & (df.fold == f)]
            avg_sigma = df_hf.sigma.mean()
            std_sigma = df_hf.sigma.std()
            print(f'Horizon {h} and Fold {f} Sigma: {round(avg_sigma, 1)} ({round(std_sigma, 1)})')

            entries['horizon'].append(h)
            entries['fold'].append(f)
            entries['sigma'].append(avg_sigma)
            entries['std'].append(std_sigma)

    pd.DataFrame(entries).to_excel('/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/avg_sigmas.xlsx')


def get_windows_with_largest_discrepancy(model_evaluation_0, model_evaluation_1, criterion='peg_loss', num_windows=10):
    loss_dfs = []
    for model_evaluation in [model_evaluation_0, model_evaluation_1]:
        inp_tar_out = model_evaluation.test_inp_tar_out
        targets = torch.tensor(
            inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'target' in col]].iloc[:, 2:].values).to(
            model_evaluation.device).unsqueeze(dim=2)
        means = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'output' in col]].values).to(
            model_evaluation.device)
        sigmas = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'sigma' in col]].values).to(
            model_evaluation.device)
        outputs = (means, sigmas)
        window_ids = inp_tar_out.window_id
        window_losses = pd.DataFrame(window_ids)
        if criterion == 'peg_loss':
            losses = torch.mean(torch.abs(torch.exp(-targets[:, :, 0] / 10) - torch.exp(-outputs[0] / 10)), dim=1)
        elif criterion == 'NLL':
            metric = nn.GaussianNLLLoss(reduction='none')
            losses = torch.mean(metric(outputs[0], targets[:, :, 0], outputs[1]), dim=1)
        losses = losses.cpu()
        window_losses['loss'] = losses
        loss_dfs.append(window_losses.copy())

    loss_df = pd.merge(loss_dfs[0], loss_dfs[1], how='inner', on='window_id')
    loss_df['diff'] = loss_df.loss_x - loss_df.loss_y

    if criterion == 'peg_loss':
        largest_discr_indices = loss_df['diff'].nlargest(n=num_windows).index
    elif criterion == 'NLL':
        # loss_df.sort_values(by='loss_y', inplace=True)
        large_diffs = loss_df.loc[loss_df['diff'] >= 1]
        largest_discr_indices = large_diffs['loss_y'].nsmallest(n=num_windows).index

    windows_of_interest_0 = model_evaluation_0.test_inp_tar_out.loc[largest_discr_indices]
    windows_of_interest_1 = model_evaluation_1.test_inp_tar_out.loc[largest_discr_indices]

    model_evaluation_0.args.plot_show = True
    model_evaluation_1.args.plot_show = True

    return windows_of_interest_0, windows_of_interest_1


def get_forecast_data(sample, model_evaluation):
    # inp_tar_out = model_evaluation.inp_tar_out
    win_id = sample.window_id.item()
    file_id = sample.file_id.item()
    window_start_dt = sample.window_start_datetime.item()

    sample_step = model_evaluation.test_data.sample_step
    train_data = model_evaluation.train_data
    val_data = model_evaluation.val_data
    test_data = model_evaluation.test_data

    window_train = train_data.window_data.loc[train_data.window_data.window_id == win_id, :]
    window_val = val_data.window_data.loc[val_data.window_data.window_id == win_id, :]
    window_test = test_data.window_data.loc[test_data.window_data.window_id == win_id, :]
    window = pd.concat(objs=[window_train, window_val, window_test], axis=0, ignore_index=True)

    slicing_train = train_data.imputation_slicing_times.loc[train_data.imputation_slicing_times.window_id == win_id, :]
    slicing_val = val_data.imputation_slicing_times.loc[val_data.imputation_slicing_times.window_id == win_id, :]
    slicing_test = test_data.imputation_slicing_times.loc[test_data.imputation_slicing_times.window_id == win_id, :]
    slicing = pd.concat(objs=[slicing_train, slicing_val, slicing_test], axis=0, ignore_index=True)

    inputs = sample.loc[:, [col for col in sample.columns if 'input' in col]].iloc[:, 2:].values[0]
    targets = sample.loc[:, [col for col in sample.columns if 'target' in col]].iloc[:, 2:].values[0]
    mus = sample.loc[:, [col for col in sample.columns if 'output' in col]].values[0]
    sigmas = sample.loc[:, [col for col in sample.columns if 'sigma' in col]].values[0]

    first_index = slicing.first_index_used_for_input_imputation.item()
    last_index = slicing.last_index_used_for_target_imputation.item()
    measure_times = [window[f'time_{ind}'].item() for ind in range(first_index, last_index)]
    measure_datetimes = [window_start_dt + datetime.timedelta(minutes=int(rel_time)) for rel_time in measure_times]
    measure_values = [window[f'glucose_value_{ind}'].item() for ind in range(first_index, last_index)]
    date = measure_datetimes[0].date()

    first_input_time = slicing.first_time_resampled_input.item()
    last_input_time = slicing.last_time_resampled_input.item()
    input_times = np.arange(first_input_time, last_input_time + 1, step=sample_step, dtype=int)
    input_datetimes = [window_start_dt + datetime.timedelta(minutes=int(rel_time)) for rel_time in input_times]
    first_target_time = slicing.first_time_resampled_target.item()
    last_target_time = slicing.last_time_resampled_target.item()
    target_times = np.arange(first_target_time, last_target_time + 1, step=sample_step, dtype=int)
    target_datetimes = [window_start_dt + datetime.timedelta(minutes=int(rel_time)) for rel_time in target_times]

    kpis = model_evaluation.calculate_model_metrics(sample)
    kpis = kpis[['loss', 'RMSE']].to_dict(orient='records')[0]

    return file_id, date, input_datetimes, inputs, target_datetimes, targets, mus, sigmas, measure_datetimes, measure_values, kpis


def load_logs(run_path):
    logs_dir_path = os.path.join(run_path, 'logs')
    log_paths = os.listdir(logs_dir_path)

    logs = {}
    for fold in range(len(log_paths)):
        log_fold_dir = [path for path in log_paths if str(fold) in path][0]
        log_fold_path = os.path.join(logs_dir_path, log_fold_dir)

        with open(log_fold_path, 'rb') as f:
            log_fold = pickle.load(f)

        for dataset in log_fold.keys():
            if 'PEG [% in A-E]' in log_fold[dataset].keys():
                log_fold[dataset]['PEG_A'] = [array[0] for array in log_fold[dataset]['PEG [% in A-E]']]
                log_fold[dataset]['PEG_B'] = [array[1] for array in log_fold[dataset]['PEG [% in A-E]']]
                log_fold[dataset]['PEG_C'] = [array[2] for array in log_fold[dataset]['PEG [% in A-E]']]
                log_fold[dataset]['PEG_D'] = [array[3] for array in log_fold[dataset]['PEG [% in A-E]']]
                log_fold[dataset]['PEG_E'] = [array[4] for array in log_fold[dataset]['PEG [% in A-E]']]
                del log_fold[dataset]['PEG [% in A-E]']
            if 'last PEG [% in A-E]' in log_fold[dataset].keys():
                log_fold[dataset]['PEG_A_last'] = [array[0] for array in log_fold[dataset]['last PEG [% in A-E]']]
                log_fold[dataset]['PEG_B_last'] = [array[1] for array in log_fold[dataset]['last PEG [% in A-E]']]
                log_fold[dataset]['PEG_C_last'] = [array[2] for array in log_fold[dataset]['last PEG [% in A-E]']]
                log_fold[dataset]['PEG_D_last'] = [array[3] for array in log_fold[dataset]['last PEG [% in A-E]']]
                log_fold[dataset]['PEG_E_last'] = [array[4] for array in log_fold[dataset]['last PEG [% in A-E]']]
                del log_fold[dataset]['last PEG [% in A-E]']

        logs[fold] = log_fold

    logs['mean'] = {dataset: {kpi: [] for kpi in logs[0][dataset].keys()} for dataset in logs[0].keys()}
    logs['std'] = {dataset: {kpi: [] for kpi in logs[0][dataset].keys()} for dataset in logs[0].keys()}
    for dataset in logs[0].keys():
        for kpi in logs[0][dataset].keys():
            ds_kpi_all_folds = [logs_fold[dataset][kpi] for fold, logs_fold in logs.items() if
                                fold not in ['mean', 'std']]
            max_len = max([len(k) for k in ds_kpi_all_folds])
            for k in ds_kpi_all_folds:
                k.extend((max_len - len(k)) * [np.nan])
            ds_kpi_all_folds = np.array(ds_kpi_all_folds)
            logs['mean'][dataset][kpi] = np.nanmean(ds_kpi_all_folds, axis=0)
            logs['std'][dataset][kpi] = np.nanstd(ds_kpi_all_folds, axis=0)

    return logs


def load_best_model_logs(run_path):
    logs = load_logs(run_path)

    for fold in logs.keys():

        best_val_loss = np.inf
        for epoch in range(1, len(logs['mean']['val']['loss'])):
            curr_val_loss = logs[fold]['val']['loss'][epoch]

            best_models_kpi_index = -1
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                best_models_kpi_index = 0

            for dataset in logs[fold].keys():
                for kpi in logs[fold][dataset].keys():
                    logs[fold][dataset][kpi][epoch] = logs[fold][dataset][kpi][epoch + best_models_kpi_index]

    logs['mean'] = {dataset: {kpi: [] for kpi in logs[0][dataset].keys()} for dataset in logs[0].keys()}
    logs['std'] = {dataset: {kpi: [] for kpi in logs[0][dataset].keys()} for dataset in logs[0].keys()}
    for dataset in logs[0].keys():
        for kpi in logs[0][dataset].keys():
            ds_kpi_all_folds = [logs_fold[dataset][kpi] for fold, logs_fold in logs.items() if
                                fold not in ['mean', 'std']]
            max_len = max([len(k) for k in ds_kpi_all_folds])
            for k in ds_kpi_all_folds:
                k.extend((max_len - len(k)) * [np.nan])
            ds_kpi_all_folds = np.array(ds_kpi_all_folds)
            logs['mean'][dataset][kpi] = np.nanmean(ds_kpi_all_folds, axis=0)
            logs['std'][dataset][kpi] = np.nanstd(ds_kpi_all_folds, axis=0)

    return logs


def plot_logs(logs):
    datasets = logs['mean'].keys()

    sns.set()
    sns.set_context('paper')
    sns.set_style('white')

    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (8, 4)

    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    markers = {'train': 'o', 'val': 'v', 'test': '*'}

    for kpi in logs['mean']['train'].keys():

        # todo: Prettify plot labels
        # todo: Select best KPIs
        # todo: Plot against each other (over and besides: maybe remove train & val to plot in one plot NLL and NLL+PEG)
        # todo: Adjust plot sizes
        # todo: move plotting to utils.plot.py to reuse for DG Layer plots
        fig, ax = plt.subplots(figsize=size)
        for dataset in datasets:
            mean_ds_kpi = np.array(logs['mean'][dataset][kpi])
            # std_ds_kpi = np.array(logs['std'][dataset][kpi])
            ax.plot(mean_ds_kpi, color=colors[dataset], marker=markers[dataset], markevery=25, ls='-', lw=1,
                    label=f"{dataset}", zorder=2)
            # ax.fill_between(range(len(mean_ds_kpi)), mean_ds_kpi - std_ds_kpi, mean_ds_kpi + std_ds_kpi, color=colors[dataset], alpha=0.2, zorder=2)

        # Formatting
        ax.grid(linewidth=0.4)
        ax.set(xlabel='Number of Epochs', ylabel=f'{kpi}')
        plt.legend(framealpha=1)

        if 'PEG_' not in kpi:
            plt.yscale('log', base=10)
        plt.xscale('log', base=2)
        # Saving, Showing & Closing
        # if save_to:
        #     # plt.savefig(save_to + f'{plot_num}' + '.pdf', bbox_inches='tight')
        #     plt.savefig(save_to + '.png', bbox_inches='tight', dpi=2400)
        plt.show()
        # plt.close()


def compare_logs(logs_0, logs_1):
    datasets = logs_0['mean'].keys()
    kpis = logs_0['mean']['test'].keys()

    sns.set()
    sns.set_context('paper')
    sns.set_style('white')

    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (8, 4)

    for kpi in kpis:
        fig, ax = plt.subplots(figsize=size)
        mean_ds_kpi_0 = np.array(logs_0['mean']['test'][kpi])
        mean_ds_kpi_1 = np.array(logs_1['mean']['test'][kpi])
        ax.plot(mean_ds_kpi_0, color='red', marker='o', markevery=25, ls='-', lw=1, label=f"NLL", zorder=2)
        ax.plot(mean_ds_kpi_1, color='green', marker='v', markevery=25, ls='-', lw=1, label=f"NLL+PEG", zorder=2)

        # Formatting
        ax.grid(linewidth=0.4)
        ax.set(xlabel='Number of Epochs', ylabel=f'{kpi}')
        plt.legend(framealpha=1)

        if kpi in ['PEG', 'RMSE', 'PEG_D', 'PEG_D_last', 'PEG_C', 'PEG_C_last']:
            plt.yscale('log')
        plt.xscale('log', base=2)
        # Saving, Showing & Closing
        # if save_to:
        #     # plt.savefig(save_to + f'{plot_num}' + '.pdf', bbox_inches='tight')
        #     plt.savefig(save_to + '.png', bbox_inches='tight', dpi=2400)
        plt.show()
        # plt.close()


#####     04_cv_kpi_changes_with_peg_weight

def get_kpis(experiment_path, horizon, peg_weights, kpi_names):
    run_dirs = os.listdir(experiment_path)
    run_dirs_horizon = [run_dir for run_dir in run_dirs if f'_hor{horizon}' in run_dir]
    run_dirs_horizon_weights = [run_dir for run_dir in run_dirs_horizon if
                                '_NLL_' in run_dir or float(run_dir.split('Surface')[-1].split('_')[0]) in peg_weights]
    peg_weights_order = [0 if '_NLL_' in run_dir else float(run_dir.split('Surface')[1].split('_i')[0]) for run_dir in
                         run_dirs_horizon_weights]
    run_dirs_horizon_weights = [run_dir for _, run_dir in sorted(zip(peg_weights_order, run_dirs_horizon_weights))]

    kpis = {}
    for run_dir in run_dirs_horizon_weights:
        peg_weight = 0 if '_NLL_' in run_dir else float(run_dir.split('Surface')[1].split('_i')[0])
        run_path = os.path.join(experiment_path, run_dir)
        kpis[peg_weight] = get_model_kpis(run_path, kpi_names)

    return kpis


def get_model_kpis(run_path, kpi_names):
    results = pd.read_excel(run_path + '/results.xlsx')

    model_kpis = {}
    for kpi_name in kpi_names:
        kpi_column = results[kpi_name]
        clean_kpi_name = kpi_name.split(' ')[-1]
        clean_kpi_name = clean_kpi_name.split('_')[-1]
        kpi_entries = kpi_column.to_dict()
        last_index = len(kpi_entries) - 1
        kpi_entries['mean'] = kpi_entries.pop(last_index-1)
        kpi_entries['std'] = kpi_entries.pop(last_index)

        if 'NLL' in kpi_name or 'RMSE' in kpi_name:
            if 'pointwise' in kpi_name:
                kpi = {key: ast.literal_eval(re.sub(r'\s+', ',', value))[-1] for key, value in kpi_entries.items()}
            else:
                kpi = kpi_entries
            model_kpis[clean_kpi_name] = kpi

        elif 'PEG' in kpi_name:
            if 'pointwise' in kpi_name:
                pct_areas = {key: value.strip('][').split('\n ')[-1].strip('][').split()[:-1] for key, value in
                             kpi_entries.items()}
            else:
                pct_areas = {key: value.strip('][').split()[:-1] for key, value in kpi_entries.items()}
            pct_a = {key: float(value[0]) for key, value in pct_areas.items()}
            pct_b = {key: float(value[1]) for key, value in pct_areas.items()}
            pct_c = {key: float(value[2]) for key, value in pct_areas.items()}
            pct_d = {key: float(value[3]) for key, value in pct_areas.items()}
            model_kpis['% in area A'] = pct_a
            model_kpis['% in area B'] = pct_b
            model_kpis['% in area C'] = pct_c
            model_kpis['% in area D'] = pct_d

    return model_kpis


def calculate_kpi_relative_changes(kpis):
    nll_kpis = kpis[0]
    kpi_names = list(nll_kpis.keys())
    folds = list(nll_kpis[kpi_names[0]].keys())
    folds.remove('mean')
    folds.remove('std')

    peg_weights = list(kpis.keys())
    peg_weights.pop(0)

    rel_changes = {}
    for peg_weight in peg_weights:
        model_rel_changes = {}
        for kpi_name in kpi_names:
            model_rel_changes[kpi_name] = {}
            # model_rel_changes[kpi_name]['mean'] = 100 * (kpis[peg_weight][kpi_name]['mean'] - nll_kpis[kpi_name]['mean']) / nll_kpis[kpi_name]['mean']
            model_rel_changes[kpi_name] = {
                fold: 100 * (kpis[peg_weight][kpi_name][fold] - nll_kpis[kpi_name][fold]) / nll_kpis[kpi_name][fold] for
                fold in folds}
            kpi_rel_changes = np.array(list(model_rel_changes[kpi_name].values()))
            # model_rel_changes[kpi_name]['mean'] = kpi_rel_changes.mean()
            model_rel_changes[kpi_name]['mean'] = 100 * (kpis[peg_weight][kpi_name]['mean'] - nll_kpis[kpi_name]['mean']) / nll_kpis[kpi_name]['mean']
            model_rel_changes[kpi_name]['std'] = kpi_rel_changes.std()
            model_rel_changes[kpi_name]['25p'] = np.percentile(a=kpi_rel_changes, q=25)
            model_rel_changes[kpi_name]['75p'] = np.percentile(a=kpi_rel_changes, q=75)

        rel_changes[peg_weight] = model_rel_changes

    return rel_changes


def plot_kpis_for_varying_peg_weight(rel_changes, seperate_plots=False, save_to=None):
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')

    matplotlib.rcParams['axes.linewidth'] = 0.5

    # colors = {'NLL': 'blue', 'RMSE': 'grey', '% in area C': 'orange', '% in area D': 'red'}
    colors = {'NLL': 'blue', 'RMSE': 'grey', '% in area A': 'green', '% in area B': 'yellowgreen',
              '% in area C': 'orange', '% in area D': 'red'}
    peg_weights = list(rel_changes.keys())
    kpis = list(rel_changes[1].keys())
    num_kpis = len(kpis)

    x = np.arange(len(peg_weights))
    peg_weights_pow_2 = [(r'$2^{%s}$' % (str(i))) for i in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]]
    width = 0.7

    if seperate_plots:
        size = (12, 12)
        fig, ax = plt.subplots(3, 2, figsize=size)
        for kpi_num, kpi in enumerate(kpis):
            kpi_means = [rel_changes[peg_weight][kpi]['mean'] for peg_weight in peg_weights]
            ind_0, ind_1 = floor(kpi_num/2), kpi_num % 2
            ax[ind_0, ind_1].bar(x, kpi_means, width, color=colors[kpi], label=kpi)
            ax[ind_0, ind_1].grid(linewidth=0.4)
            ax[ind_0, ind_1].set_xticks(x, peg_weights_pow_2)
            ax[ind_0, ind_1].legend(framealpha=1, loc='upper left')
        ax[0, 0].set(ylabel=f'mean relative change [%]')
        ax[1, 0].set(ylabel=f'mean relative change [%]')
        ax[2, 0].set(ylabel=f'mean relative change [%]')
        ax[2, 0].set(xlabel=r'$\lambda_{PEG}$')
        ax[2, 1].set(xlabel=r'$\lambda_{PEG}$')
    else:
        size = (6, 4)
        fig, ax = plt.subplots(figsize=size)
        for kpi_num, kpi in enumerate(kpis):
            kpi_means = [rel_changes[peg_weight][kpi]['mean'] for peg_weight in peg_weights]
            kpi_stds = [rel_changes[peg_weight][kpi]['std'] for peg_weight in peg_weights]
            kpi_25ps = [rel_changes[peg_weight][kpi]['25p'] for peg_weight in peg_weights]
            kpi_75ps = [rel_changes[peg_weight][kpi]['75p'] for peg_weight in peg_weights]
            kpi_errors = np.row_stack((kpi_25ps, kpi_75ps))

            # bar = ax.bar(x - ((num_kpis - 1) / 2 - kpi_num) * width, kpi_means, width, yerr=kpi_stds, align='center', color=colors[kpi], label=kpi)
            # bar = ax.bar(x - ((num_kpis - 1) / 2) * width + kpi_num * width, kpi_means, width, yerr=kpi_stds, color=colors[kpi], label=kpi)
            # bar = ax.bar(x - ((num_kpis - 1) / 2) * width + kpi_num * width, kpi_means, width, yerr=kpi_errors, color=colors[kpi], label=kpi)
            bar = ax.bar(x - ((num_kpis - 1) / 2) * width + kpi_num * width, kpi_means, width, color=colors[kpi], label=kpi)
            # ax.bar_label(bar, padding=3)

            ax.grid(linewidth=0.4)
            ax.set(xlabel=r'$\lambda_{PEG}$', ylabel=f'mean relative change [%]')
            ax.set_xticks(x, peg_weights_pow_2)
            plt.legend(framealpha=1)

    # Formatting
    plt.tight_layout()

    # Saving, Showing & Closing
    if save_to:
        plt.savefig(save_to + '/peg_lambda_variation_kpis.png', bbox_inches='tight', dpi=1200)
    plt.show()


#####     13_tSNE
def t_SNE_main(experiment_path, gpu, save_to, model_type, domain_split, seed, horizon, fold, metric, n_iter, perplexity, lr,
               plot_gdu_bases, plot_fe_output, num_fe_outputs, plot_num_fe_outputs):
    run_path = get_run_path(experiment_path, model_type, seed, horizon)
    vectors = get_vectors(run_path, model_type, horizon, fold, seed, gpu, gdu_bases=plot_gdu_bases,
                          fe_output=plot_fe_output, max_fe_outputs=num_fe_outputs)
    embeddings = get_tSNE(vectors, perplexity, lr, n_iter, model_type, seed, horizon, fold, metric, num_fe_outputs)
    embeddings_categorized = categorize_embeddings(embeddings, domain_split, plot_gdu_bases, plot_fe_output)
    save_dir = save_to + f"{model_type}/horizon{horizon}_seed{seed}_fold{fold}/{metric}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_to = os.path.join(save_dir, f"{domain_split}_FEoutputs{num_fe_outputs}_plot{plot_num_fe_outputs}_perpl{perplexity}_lr{lr}.png")
    plot_embeddings(embeddings_categorized, plot_num_fe_outputs, save_to=save_to)


def create_t_SNE_plots(experiment_path, gpu, save_to, model_type, domain_split, seed, horizon, fold, metric, n_iter, perplexities, lrs, plot_gdu_bases, plot_fe_output, num_fe_outputs, plot_num_fe_outputs):
    # Get vectors
    run_path = get_run_path(experiment_path, model_type, seed, horizon)
    vectors = get_vectors(run_path, model_type, horizon, fold, seed, gpu, gdu_bases=plot_gdu_bases, fe_output=plot_fe_output, max_fe_outputs=num_fe_outputs)

    # Calculate embeddings
    embeddings = {perpl: {lr: {} for lr in lrs} for perpl in perplexities}
    embeddings_categorized = {perpl: {lr: {} for lr in lrs} for perpl in perplexities}
    for perplex_num, perplexity in enumerate(perplexities):
        for lr_num, lr in enumerate(lrs):
            embeddings[perplexity][lr] = get_tSNE(vectors, perplexity, lr, n_iter, model_type, seed, horizon, fold, metric, num_fe_outputs)
            embeddings_categorized[perplexity][lr] = categorize_embeddings(embeddings[perplexity][lr], domain_split, plot_gdu_bases, plot_fe_output)

    # Create plot
    save_dir = save_to + f"{model_type}/horizon{horizon}_seed{seed}_fold{fold}/{metric}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_to = os.path.join(save_dir, f"{domain_split}_FEoutputs{num_fe_outputs}_plot{plot_num_fe_outputs}_perpl{perplexities}_lr{lrs}.png")
    plot_all_embeddings(embeddings_categorized, perplexities, lrs, plot_num_fe_outputs, save_to=save_to)


def try_tsne_params(experiment_path, gpu, save_to, model_type, domain_split, seed, horizon, fold, metric, n_iter, perplexities,
                    lrs, plot_gdu_bases, plot_fe_output, num_fe_outputs, plot_num_fe_outputs):
    run_path = get_run_path(experiment_path, model_type, seed, horizon)
    vectors = get_vectors(run_path, model_type, horizon, fold, seed, gpu, gdu_bases=plot_gdu_bases,
                          fe_output=plot_fe_output, max_fe_outputs=num_fe_outputs)

    for perplexity in perplexities:
        for lr in lrs:
            plot_info = {'perplexity': perplexity, 'lr': lr}
            print('\n\n', plot_info, '\n')
            embeddings = get_tSNE(vectors, perplexity, lr, n_iter, model_type, seed, horizon, fold, metric,
                                  num_fe_outputs)
            embeddings_categorized = categorize_embeddings(embeddings, domain_split, plot_gdu_bases, plot_fe_output)
            save_to_iteration = save_to + f"model_type_{model_type}_seed_{seed}_horizon_{horizon}_fold_{fold}_metric_{metric}_perplexity{perplexity}_lr_{lr}.png"
            plot_embeddings(embeddings_categorized, plot_num_fe_outputs, save_to=save_to_iteration, plot_info=plot_info)


def get_run_path(base_path, model_type, seed, horizon):
    if base_path.endswith('Layer_Experiment'):
        if model_type == 'single':
            run_path = os.path.join(base_path, 'LSTM_single')
        elif model_type == 'ensemble':
            run_path = os.path.join(base_path, 'LSTM_ensemble')
        elif model_type in ['E2E_MMD', 'FT_MMD', 'E2E_CS', 'FT_CS']:
            run_path = os.path.join(base_path, 'LSTM_Layer', model_type)
    else:
        run_path = os.path.join(base_path, model_type)

    seed_dir_names = os.listdir(run_path)
    seed_dir_name = [dir_name for dir_name in seed_dir_names if f"_{seed}_" in dir_name or f"seed_{seed}" in dir_name][0]
    run_path = os.path.join(run_path, seed_dir_name)

    horizon_dir_names = os.listdir(run_path)
    horizon_dir_name = [dir_name for dir_name in horizon_dir_names if f"_{seed}_{horizon}_" in dir_name or f"horizon_{horizon}_" in dir_name][0]
    run_path = os.path.join(run_path, horizon_dir_name)

    return run_path


def get_vectors(run_path, model_type, horizon, fold, seed, gpu, gdu_bases=True, fe_output=True, max_fe_outputs=None):
    vectors_dir = os.path.join('results/vectors',
                               f'{model_type}_horizon{horizon}_fold{fold}_seed{seed}_gdus{gdu_bases}_fes{fe_output}_maxFEoutput{max_fe_outputs}')

    if os.path.exists(vectors_dir):
        vectors = pd.read_pickle(vectors_dir + '/vectors.pkl')

    else:
        vectors = []
        if gdu_bases:
            gdu_base_vectors = get_gdu_bases(run_path, fold)
            vectors.extend(gdu_base_vectors)
        if fe_output:
            fe_output_vectors = get_fe_output(run_path=run_path, model_num=fold, gpu=gpu, max_output=max_fe_outputs)
            vectors.extend(fe_output_vectors)
        vectors = pd.DataFrame(vectors)

        os.makedirs(vectors_dir)
        vectors.to_pickle(vectors_dir + '/vectors.pkl')
    return vectors


def get_gdu_bases(run_path, model_num):
    # Initialize
    with open(run_path + '/args.json') as file:
        args = json.load(file)
    args = dotdict(args)
    # args.base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
    if args.Layer_hps['fine_tuning_FEs_path']:
        args.Layer_hps['fine_tuning_FEs_path'] = os.path.join(
            # '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment',
            '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression',
            args.Layer_hps['fine_tuning_FEs_path'])
    sys.path.append(os.path.expanduser(args.repo_location))
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cpu'

    args.data_path = '/local/home/ansass/Thesis/icarus/analysis/data_on_server/v_2/4_imputed_data/' + args.data_path.split('/', -1)[-1] # todo: only works for ansass
    data_path = args.data_path
    # Get Data and Model
    train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        model_num, data_path, args)
    model, optimiser, scheduler = get_model(args, device, train_data, train_dataset,
                                            num_input_features=train_data[0][0].size(1),
                                            input_length=train_data[0][0].size(0),
                                            output_size=train_data[0][-1].size(0), fold=model_num)
    model.load_state_dict(torch.load(os.path.join(run_path, "models", f"weights_{model_num}.pth")))

    gdu_bases = {}
    for name, params in model.gdu_layer.gdus.named_parameters():
        if params.requires_grad:
            gdu_name = 'V_' + name.split('.')[0].split('_')[1]
            bases_tensor = np.transpose(np.squeeze(params.data.numpy()))
            gdu_bases[gdu_name] = bases_tensor

    gdu_names = list(gdu_bases.keys())
    base_size = gdu_bases[gdu_names[0]].shape[0]
    vec_len = gdu_bases[gdu_names[0]].shape[1]

    gdu_base_vectors = []
    for gdu_name in gdu_names:
        gdu_base = gdu_bases[gdu_name]
        for vector_num in range(base_size):
            vec_entries = {f'vec_{i}': gdu_base[vector_num, i] for i in range(vec_len)}
            vector = {'gdu': gdu_name, **vec_entries}
            gdu_base_vectors.append(vector)
    return gdu_base_vectors


def get_fe_output(run_path, model_num, gpu, max_output=None):
    fe_output_dir = os.path.join('results/fe_outputs', run_path.split('Thesis/')[-1], f'model_num_{model_num}')
    if os.path.exists(fe_output_dir):
        in_out = pd.read_pickle(fe_output_dir + '/all_folds.pkl')
    else:
        in_out = calc_fe_output(run_path=run_path, fold=model_num, gpu=gpu, data='all_folds')
        output_columns = [col for col in in_out.columns if 'output_' in col]
        in_out = in_out.rename(columns={out_col: f'vec_{i}' for i, out_col in enumerate(output_columns)})
        os.makedirs(fe_output_dir)
        in_out.to_pickle(fe_output_dir + '/all_folds.pkl')

    in_out_by_treatment = categorize_data_by_domains(in_out, domain_split='treatment', max_output=max_output)
    in_out_sample = pd.concat(objs=in_out_by_treatment, axis=0, ignore_index=True)
    in_out_vectors = in_out_sample.to_dict('records')
    return in_out_vectors


def categorize_data_by_domains(in_out, domain_split, max_output=None):
    domain_split_criterium = domain_split if 'glucose_level' not in domain_split else 'glucose_level'
    domains = list(DOMAINS[domain_split_criterium].items())

    data = {}
    for domain_name, sub_domains in domains:
        in_out_domain = filter_domain_data(in_out, sub_domains=sub_domains, domain_split=domain_split)
        # num_fe_outputs = len(in_out)
        # num_fe_outputs_domain = len(in_out_domain)
        # num_samples_for_run = round(max_output * num_fe_outputs_domain / num_fe_outputs)
        # in_out_domain = in_out_domain if max_output is None or max_output >= len(in_out_domain) else in_out_domain.sample(n=num_samples_for_run)
        in_out_domain = in_out_domain if max_output is None or max_output >= len(in_out_domain) else in_out_domain.sample(n=max_output)
        data[domain_name] = in_out_domain
    return data


def filter_domain_data(in_out, sub_domains, domain_split='treatment'):
    if domain_split in ['treatment', 'diabetes_type']:
        in_out_domain = in_out.loc[in_out[domain_split].isin(sub_domains), :]
    elif domain_split in ['HbA1c_level', 'glucose_level_last_input']:
        criterium_column_name = [colname for colname in in_out.columns if 'input_' in colname][-1]
        in_out_domain = in_out.loc[
                        in_out[criterium_column_name].between(sub_domains[0], sub_domains[1], inclusive='left'), :]
    elif domain_split == 'slope':
        glucose_column_names = [colname for colname in in_out.columns if 'input_' in colname][-2:]
        slope_column = in_out[glucose_column_names[-1]] - in_out[glucose_column_names[-2]]
        in_out['slope'] = slope_column
        in_out_domain = in_out.loc[in_out['slope'].between(sub_domains[0], sub_domains[1], inclusive='left'), :]
    return in_out_domain


def get_tSNE(vectors, perplexity, lr, n_iter, model_type, seed, horizon, fold, metric, max_fe_output, sigma=9.5):
    vector_columns = [col for col in vectors.columns if 'vec_' in col]
    vector_entries = vectors[vector_columns].to_numpy()
    embeddings = calc_tSNE(vector_entries, perplexity, lr, n_iter, model_type, seed, horizon, fold, metric,
                           max_fe_output, sigma)
    vectors[['x', 'y']] = embeddings
    return vectors


def categorize_embeddings(vectors, domain_split, plot_gdu_bases, plot_fe_output):
    vectors_categorized = {}
    if plot_gdu_bases:
        gdu_names = vectors.gdu.dropna().unique()
        gdu_vectors_categorized = {gdu_name: vectors.loc[vectors.gdu == gdu_name] for gdu_name in gdu_names}
        vectors_categorized.update(gdu_vectors_categorized)
    if plot_fe_output:
        fe_output_categorized = categorize_data_by_domains(vectors, domain_split)
        vectors_categorized.update(fe_output_categorized)

    embeddings_categorized = {}
    for domain, in_out_domain in vectors_categorized.items():
        embeddings = in_out_domain[['x', 'y']].to_numpy(copy=True)
        embeddings_categorized[domain] = embeddings
    return embeddings_categorized


def categorize_gdu_vectors(vectors):
    gdu_names = vectors.gdu.unique()

    gdu_vectors = {gdu_name: vectors.loc[vectors.gdu == gdu_name] for gdu_name in vectors.gdu.unique()}
    for gdu_name in gdu_names:
        gdu_vectors[gdu_name] = vectors.loc[vectors.gdu == gdu_name]
    return gdu_vectors


def calc_tSNE(vectors, perplexity, lr, n_iter, model_type, seed, horizon, fold, metric, max_fe_output, sigma):
    tsne_dir = os.path.join('results/tsne',
                            f'{model_type}_{metric}_seed{seed}_horizon{horizon}_fold{fold}_maxFEoutput{max_fe_output}_perplexity{perplexity}_lr{lr}_niter{n_iter}')
    if os.path.exists(tsne_dir):
        points = np.load(tsne_dir + '/tSNE.npy')
    else:
        if metric == 'euclidean':
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, n_iter=n_iter, init='pca', verbose=1,
                        metric=metric)
        elif metric == 'MMD':
            metric = 'precomputed'
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, n_iter=n_iter, init='random',
                        verbose=1, metric=metric)
            vectors = calc_pairwise_mmd(vectors)
        points = tsne.fit_transform(vectors)
        os.makedirs(tsne_dir)
        np.save(tsne_dir + '/tSNE', points)
    return points


def calc_pairwise_mmd(vectors, sigma=9.5):
    gamma = 1 / (2 * sigma ** 2)
    K = metrics.pairwise.rbf_kernel(vectors, gamma=gamma)
    distance = 2 - 2 * K
    return distance


def plot_embeddings(embeddings, plot_num_fe_outputs, save_to=None, plot_info=None):
    categories = list(embeddings.keys())
    num_categories = len(categories)
    gdu_bases = [category for category in categories if 'V_' in category]
    num_gdu_bases = len(gdu_bases)
    fe_output_clusters = [category for category in categories if 'V_' not in category]
    num_fe_output_clusters = len(fe_output_clusters)

    num_fe_points = sum([len(points) for category, points in embeddings.items() if category in fe_output_clusters])

    if plot_num_fe_outputs != 'all':
        for fe_output_cluster in fe_output_clusters:
            fe_cluster = embeddings[fe_output_cluster]
            num_samples_category = round(plot_num_fe_outputs * len(fe_cluster) / num_fe_points)
            embeddings[fe_output_cluster] = fe_cluster[
                np.random.choice(fe_cluster.shape[0], num_samples_category, replace=False)]

    # Define colors, markers and sizes
    colors = {}
    colors_gdu_bases = cm.rainbow(np.linspace(0, 0.6, num_gdu_bases))
    colors_gdu_bases = {gdu_base: color for gdu_base, color in zip(gdu_bases, colors_gdu_bases)}
    colors.update(colors_gdu_bases)
    colors_fe_output = cm.rainbow(np.linspace(0.7, 1, num_fe_output_clusters))
    colors_fe_output = {fe_output: color for fe_output, color in zip(fe_output_clusters, colors_fe_output)}
    colors.update(colors_fe_output)

    markers = {}
    markers_gdu_bases = ['o', '^', 's', 'p', 'v', 'P', 'D', '<', 'H']
    markers_gdu_bases = {gdu_base: marker for gdu_base, marker in zip(gdu_bases, markers_gdu_bases)}
    markers.update(markers_gdu_bases)
    markers_fe_outputs = ['h', '>', 'X', 'd', '*', '1', '|', '2']
    markers_fe_outputs = {fe_output: marker for fe_output, marker in zip(fe_output_clusters, markers_fe_outputs)}
    markers.update(markers_fe_outputs)

    sizes = {}
    sizes_gdu_bases = {gdu_base: 50 for gdu_base in gdu_bases}
    sizes_fe_outputs = {fe_output: 30 for fe_output in fe_output_clusters}
    sizes.update(sizes_gdu_bases)
    sizes.update(sizes_fe_outputs)

    zorders = {}
    zorders_gdu_bases = {gdu_base: 5 for gdu_base in gdu_bases}
    zorders_fe_outputs = {fe_output: 4 for fe_output in fe_output_clusters}
    zorders.update(zorders_gdu_bases)
    zorders.update(zorders_fe_outputs)

    labels = {}
    labels_gdu_bases = {gdu_base: rf"${gdu_base}$" for gdu_base in gdu_bases}
    labels_fe_outputs = {fe_output: fe_output for fe_output in fe_output_clusters}
    labels.update(labels_gdu_bases)
    labels.update(labels_fe_outputs)

    # Plotting
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (6, 6)
    fig, ax = plt.subplots(figsize=size)
    for category in categories:
        x, y = embeddings[category][:, 0], embeddings[category][:, 1]
        ax.scatter(x, y, color=colors[category], edgecolors='white', marker=markers[category], s=sizes[category],
                   zorder=zorders[category], label=labels[category])
    # ax.grid(linewidth=0.4)
    ax.grid(False)
    #plt.legend(framealpha=1) todo
    if plot_info:
        ax.set_title(json.dumps(plot_info))

    # Saving, Showing & Closing
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_all_embeddings(embeddings, perplexities, lrs, plot_num_fe_outputs, save_to=None, plot_info=None):
    perpl_0, lr_0 = perplexities[0], lrs[0]
    categories = list(embeddings[perpl_0][lr_0].keys())
    num_categories = len(categories)
    gdu_bases = [category for category in categories if 'V_' in category]
    num_gdu_bases = len(gdu_bases)
    fe_output_clusters = [category for category in categories if 'V_' not in category]
    num_fe_output_clusters = len(fe_output_clusters)

    if plot_num_fe_outputs != 'all':
        for perplex_num, perplexity in enumerate(perplexities):
            for lr_num, lr in enumerate(lrs):
                for fe_output_cluster in fe_output_clusters:
                    num_fe_points = sum([len(points) for category, points in embeddings[perplexity][lr].items() if category in fe_output_clusters])
                    fe_cluster = embeddings[perplexity][lr][fe_output_cluster]
                    num_samples_category = round(plot_num_fe_outputs * len(fe_cluster) / num_fe_points)
                    embeddings[perplexity][lr][fe_output_cluster] = fe_cluster[np.random.choice(fe_cluster.shape[0], num_samples_category, replace=False)]

    # Define colors, markers and sizes
    colors = {}
    colors_gdu_bases = cm.rainbow(np.linspace(0, 0.6, num_gdu_bases))
    colors_gdu_bases = {gdu_base: color for gdu_base, color in zip(gdu_bases, colors_gdu_bases)}
    colors.update(colors_gdu_bases)
    colors_fe_output = cm.rainbow(np.linspace(0.7, 1, num_fe_output_clusters))
    colors_fe_output = {fe_output: color for fe_output, color in zip(fe_output_clusters, colors_fe_output)}
    colors.update(colors_fe_output)

    markers = {}
    markers_gdu_bases = ['o', '^', 's', 'p', 'v', 'P', 'D', '<', 'H']
    markers_gdu_bases = {gdu_base: marker for gdu_base, marker in zip(gdu_bases, markers_gdu_bases)}
    markers.update(markers_gdu_bases)
    markers_fe_outputs = ['h', '>', 'X', 'd', '*', '1', '|', '2']
    markers_fe_outputs = {fe_output: marker for fe_output, marker in zip(fe_output_clusters, markers_fe_outputs)}
    markers.update(markers_fe_outputs)

    sizes = {}
    sizes_gdu_bases = {gdu_base: 50 for gdu_base in gdu_bases}
    sizes_fe_outputs = {fe_output: 30 for fe_output in fe_output_clusters}
    sizes.update(sizes_gdu_bases)
    sizes.update(sizes_fe_outputs)

    zorders = {}
    zorders_gdu_bases = {gdu_base: 5 for gdu_base in gdu_bases}
    zorders_fe_outputs = {fe_output: 4 for fe_output in fe_output_clusters}
    zorders.update(zorders_gdu_bases)
    zorders.update(zorders_fe_outputs)

    labels = {}
    labels_gdu_bases = {gdu_base: rf"${gdu_base}$" for gdu_base in gdu_bases}
    labels_fe_outputs = {fe_output: fe_output for fe_output in fe_output_clusters}
    labels.update(labels_gdu_bases)
    labels.update(labels_fe_outputs)

    # Plotting
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (14.5, 17)
    fig, axs = plt.subplots(len(perplexities), len(lrs), figsize=size)
    for perplex_num, perplexity in enumerate(perplexities):
        for lr_num, lr in enumerate(lrs):
            for category in categories:
                x, y = embeddings[perplexity][lr][category][:, 0], embeddings[perplexity][lr][category][:, 1]
                axs[perplex_num, lr_num].scatter(x, y, color=colors[category], edgecolors='white', marker=markers[category], s=sizes[category], zorder=zorders[category], label=labels[category])
            axs[perplex_num, lr_num].grid(False)
            axs[perplex_num, lr_num].xaxis.set_tick_params(labelbottom=False)
            axs[perplex_num, lr_num].yaxis.set_tick_params(labelleft=False)
            axs[perplex_num, lr_num].set_xticks([])
            axs[perplex_num, lr_num].set_yticks([])
            # axs[perplex_num, lr_num].axis('off')
            axs[perplex_num, lr_num].set(xlabel=f"learning rate: {lr}", ylabel=f"perplexity: {perplexity}")
            axs[perplex_num, lr_num].label_outer()
    all_handles, all_labels = axs[0, 0].get_legend_handles_labels()
    gdu_handles, fv_handles = all_handles[:num_gdu_bases], all_handles[num_gdu_bases:]
    gdu_labels, fv_labels = all_labels[:num_gdu_bases], all_labels[num_gdu_bases:]
    fv_legend = fig.legend(fv_handles, fv_labels, loc='lower right', bbox_to_anchor=(0.504, 0.9), framealpha=1)
    gdu_legend = fig.legend(gdu_handles, gdu_labels, loc='lower left', bbox_to_anchor=(0.52, 0.9), ncol=3, framealpha=1)
    # plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.2), ncol=len(labels)-5, framealpha=1)
    # plt.tight_layout()

    # todo: have two legends: one with gdu bases and one with feature vector classification
    # todo: position the legends on top of the plot and center them in the right way independent from the last subfigure
    # todo: check if tight_layout contributes to the difference in size between heatmap and tsne plots

    # Saving, Showing & Closing
    if save_to:
        # plt.savefig(save_to, bbox_extra_artists=(fv_legend, gdu_legend), bbox_inches='tight', dpi=600)
        plt.savefig(save_to, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def calculate_model_outputs(model, data, model_num):
    if type(data) is not list:
        data = [data]

    in_out_dfs = []
    for data_split in data:
        model.eval()
        inputs, _ = data_split[:]
        inputs = inputs.float().to(model.device)
        with torch.no_grad():
            x = model(inputs)
        x = x.cpu().detach().numpy()
        len_output = x.shape[1]

        inputs, _ = data_split[:]
        glucose_input = inputs[:, :, 0]
        len_input = glucose_input.size()[1]
        num_samples = glucose_input.size()[0]
        inp_out_data = np.concatenate((glucose_input, x), axis=1)
        column_names = [f'input_{i}' for i in range(len_input)] + [f'output_{i}' for i in range(len_output)]
        data_df = pd.DataFrame(inp_out_data, columns=column_names)
        info_df = pd.DataFrame([model_num] * num_samples, columns=['fold'])
        file_data_df = data_split.window_data[
            ['window_id', 'file_id', 'study', 'patient_id_study', 'sensor', 'treatment', 'diabetes_type',
             'diabetes_duration', 'hba1c', 'sex', 'age', 'window_start_datetime']]
        all_data_df = pd.concat([info_df,
                                 file_data_df,
                                 data_split.imputation_slicing_times[
                                     ['first_dt_resampled_input', 'last_dt_resampled_input',
                                      'first_dt_resampled_target',
                                      'last_dt_resampled_target']],
                                 data_df], axis=1)
        all_data_df.rename(columns={'first_dt_resampled_input': 'first_input_dt',
                                    'last_dt_resampled_input': 'last_input_dt',
                                    'first_dt_resampled_target': 'first_target_dt',
                                    'last_dt_resampled_target': 'last_target_dt'}, inplace=True)

        in_out_dfs.append(all_data_df)
    in_out = pd.concat(objs=in_out_dfs, axis=0, ignore_index=True)
    return in_out


#####     08 DG Exemplary Forecasts

class ModelLoaderEvaluator:
    def __init__(
            self,
            base_path,
            model_type,
            seed,
            horizon,
            fold,
            gpu,

    ):
        self.model_type = model_type
        self.seed = seed
        self.horizon = horizon
        self.fold = fold
        self.gpu = gpu

        self.model = None
        self.evaluation = None
        self.args = None
        self.logs = None
        self.device = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.run_path = get_run_path(base_path, model_type, seed, horizon)
        self.load_model()
        self.evaluate_model()

    def load_model(self):
        print(f'\n Loading model at: {self.run_path}')
        # Initialize Evaluation with same args and data as original run. Adapt save string, but keep load string for model loading
        with open(self.run_path + '/args.json') as file:
            self.args = json.load(file)
            self.args = dotdict(self.args)
        with open(self.run_path + '/gp_args.json') as file:
            gp_args = json.load(file)
            gp_args = dotdict(gp_args)

        self.args.base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
        sys.path.append(os.path.expanduser(self.args.repo_location))
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        self.device = init_gpu(self.gpu)
        data_path = self.args.data_path

        self.args.save_string = self.run_path + '/evaluations/' + time.strftime('%Y%m%d-%H%M%S')
        self.args.domain = 'treatment'
        self.args.plot_show = False
        if not os.path.exists(self.args.save_string):
            os.makedirs(self.args.save_string)

        with open(os.path.join(self.run_path, 'logs.pkl'), 'rb') as f:
            self.logs = pickle.load(f)

        print("\n Loading Data Split: ", self.fold)
        # load data, model & metrics
        self.train_data, self.val_data, self.test_data, train_loader, val_loader, test_loader, self.train_dataset, self.val_dataset, self.test_dataset = load_data(
            self.fold, data_path, self.args)
        self.model, optimiser, scheduler = get_model(self.args, self.device, self.train_data, self.train_dataset,
                                                     num_input_features=self.train_data[0][0].size(1),
                                                     input_length=self.train_data[0][0].size(0),
                                                     output_size=self.train_data[0][-1].size(0), fold=self.fold)
        self.model.load_state_dict(torch.load(os.path.join(self.run_path, "models", f"weights_{self.fold}.pth")))
        del train_loader, val_loader, test_loader, optimiser, scheduler
        torch.cuda.empty_cache()

    def evaluate_model(self):
        criterion, metrics = get_metrics(self.args, self.train_data, self.device)
        self.evaluation = ModelEvaluator(self.model,
                                         self.train_data, self.val_data, self.test_data,
                                         self.train_dataset, self.val_dataset, self.test_dataset,
                                         criterion, metrics, self.args, self.fold, self.logs)


#####     11 Heatmap: KPI changes with varying GDU loss weights

def get_gdu_kpis(experiment_path, ols_weights, l1_weights, kpi_names):
    similarities = ['CS', 'MMD']
    seeds = [0, 1, 2, 3, 4]
    folds = [0, 1, 2, 3]

    # get all similarity and seed directories
    all_dirs = os.listdir(experiment_path)
    all_similarity_seed_dirs = [directory for directory in all_dirs if f'Layers_LSTM_' in directory]

    # get clean kpi names
    eg_sim_seed_path = os.path.join(experiment_path, all_similarity_seed_dirs[0])
    eg_run_path = os.path.join(eg_sim_seed_path, [d for d in os.listdir(eg_sim_seed_path) if 'OLS' in d][0])
    clean_kpi_names = get_model_kpis(eg_run_path, kpi_names).keys()

    # create kpi structure
    kpis = {
        similarity: {
            seed: {
                clean_kpi_name: {
                    fold: pd.DataFrame(columns=l1_weights, index=ols_weights, dtype=float) for fold in folds
                } for clean_kpi_name in clean_kpi_names
            } for seed in itertools.chain(seeds, ['mean'])
        } for similarity in similarities
    }

    # fill the kpi structure
    for similarity in similarities:
        for seed in seeds:
            sim_seed_dir = [directory for directory in all_similarity_seed_dirs if f'_{similarity}_' in directory and f'_{seed}_' in directory][0]
            sim_seed_path = os.path.join(experiment_path, sim_seed_dir)
            sim_seed_all_dirs = os.listdir(sim_seed_path)
            sim_seed_run_dirs = [directory for directory in sim_seed_all_dirs if f'_treatment_' in directory]

            for ols_weight in ols_weights:
                for l1_weight in l1_weights:
                    run_dir = [d for d in sim_seed_run_dirs if f'lambdaOLS' + ols_weight in d and f'lambda_l1' + l1_weight in d][0]
                    run_path = os.path.join(sim_seed_path, run_dir)
                    run_kpis = get_model_kpis(run_path, kpi_names)

                    for clean_kpi_name in clean_kpi_names:
                        for fold in folds:
                            kpis[similarity][seed][clean_kpi_name][fold][l1_weight][ols_weight] = run_kpis[clean_kpi_name][fold] # todo: copy?

    # aggregate kpis over seeds
    for clean_kpi_name in clean_kpi_names:
        for similarity in similarities:
            for fold in folds:
                kpis[similarity]['mean'][clean_kpi_name][fold] = (kpis[similarity][0][clean_kpi_name][fold] +
                                                                  kpis[similarity][1][clean_kpi_name][fold] +
                                                                  kpis[similarity][2][clean_kpi_name][fold] +
                                                                  kpis[similarity][3][clean_kpi_name][fold] +
                                                                  kpis[similarity][4][clean_kpi_name][fold]) / 5

    return kpis


def heatmap_kpis_for_varying_gdu_weights(kpis, save_to):
    similarity_names = list(kpis.keys())
    kpi_names = list(kpis[similarity_names[0]]['mean'].keys())
    folds = list(kpis[similarity_names[0]]['mean'][kpi_names[0]].keys())
    num_sims = len(similarity_names)

    xticklabels = [str(weight) for weight in list(kpis[similarity_names[0]]['mean'][kpi_names[0]][folds[0]].columns)]
    yticklabels = [str(weight) for weight in list(kpis[similarity_names[0]]['mean'][kpi_names[0]][folds[0]].index)]
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (6, 6)

    for fold in folds:
    # for fold in [1]:
        for kpi_name in kpi_names:
        # for kpi_name in ['NLL']:
            for num_sim, similarity_name in enumerate(similarity_names):
                fig, ax = plt.subplots(figsize=size)
                sns.heatmap(kpis[similarity_name]['mean'][kpi_name][fold], xticklabels=xticklabels, yticklabels=yticklabels, annot=True, fmt='.4g', square=True, cbar_kws={'shrink': 0.8}, ax=ax)
                # ax.set(xlabel=r'$\lambda_{L_1}$', ylabel=r'$\lambda_{OLS}$', title=f'{kpi_name} | {similarity_name} | fold: {fold}')
                ax.set(xlabel=r'$\lambda_{L_1}$', ylabel=r'$\lambda_{OLS}$')
                # Saving, Showing & Closing
                if save_to:
                    plt.savefig(save_to + f'{kpi_name}_{similarity_name}_fold_{fold}.png', bbox_inches='tight', dpi=600)
                plt.show()
                plt.close()


#####     12 Similarities between domains and bases

def categorize_vectors(raw_vectors, domain_split):
    vectors_categorized = {}

    # GDU bases
    gdu_names = raw_vectors.gdu.dropna().unique()
    gdu_vectors_categorized = {gdu_name: raw_vectors.loc[raw_vectors.gdu == gdu_name] for gdu_name in gdu_names}
    vectors_categorized.update(gdu_vectors_categorized)

    # FE output
    fe_output_categorized = categorize_data_by_domains(raw_vectors, domain_split)
    vectors_categorized.update(fe_output_categorized)

    # Clean the vectors to be just the vector entries and no other information. Cast to torch tensor.
    vec_cols = [col for col in raw_vectors.columns if col.startswith('vec_')]
    for category, in_out in vectors_categorized.items():
        vec_entries = in_out[vec_cols].values
        vectors_categorized[category] = np.array(vec_entries)

    return vectors_categorized


def get_similarities(vectors, model_type, horizon, fold, seed, domain_split):
    save_path = os.path.join('results/12_similarity_heatmap',
                             f'{model_type}_horizon{horizon}_fold{fold}_seed{seed}_split{domain_split}')

    if os.path.exists(save_path):
        similarities_df = pd.read_pickle(save_path + '/similarities.pkl')

    else:
        metric = 'CS' if 'CS' in model_type else 'MMD'
        category_names = list(vectors.keys())
        similarities_df = pd.DataFrame(columns=category_names, index=category_names)
        for category_0 in category_names:
            for category_1 in category_names:
                print(f'Calculate {metric} between {category_0} and {category_1}...')
                similarities_df.loc[category_0, category_1] = calc_similarity(vectors[category_0], vectors[category_1],
                                                                              metric=metric)
        similarities_df = similarities_df.astype(float)

        os.makedirs(save_path)
        similarities_df.to_pickle(save_path + '/similarities.pkl')

    return similarities_df, save_path


def calc_similarity(vectors_0, vectors_1, metric):
    sigma = 9.5
    gamma = 1 / (2 * sigma ** 2)
    num_vectors_0 = vectors_0.shape[0]
    num_vectors_1 = vectors_1.shape[0]

    # Calculate Gram-Matrix K
    K_00 = metrics.pairwise.rbf_kernel(vectors_0, vectors_0, gamma)
    K_01 = metrics.pairwise.rbf_kernel(vectors_0, vectors_1, gamma)
    K_11 = metrics.pairwise.rbf_kernel(vectors_1, vectors_1, gamma)

    K_00 = K_00.sum() * 1 / (num_vectors_0 * num_vectors_0)
    K_01 = K_01.sum() * 1 / (num_vectors_0 * num_vectors_1)
    K_11 = K_11.sum() * 1 / (num_vectors_1 * num_vectors_1)

    if metric == 'CS':
        similarity = K_01 / (K_00 * K_11) ** (1 / 2)
    else:
        similarity = - (K_00 - 2 * K_01 + K_11)

    return similarity


def create_similarity_heatmap(similarities, save_to=None):
    categories = [cat if 'V_' not in cat else rf'${cat}$' for cat in similarities.columns]
    mask = np.triu(similarities.to_numpy())

    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (6, 6)

    plt.figure(figsize=size)
    sns.heatmap(similarities.round(2), mask=mask, xticklabels=categories, yticklabels=categories, annot=True, annot_kws={"fontsize": 7}, square=True, cbar_kws={'shrink': 0.8})
    plt.tight_layout()

    # Saving, Showing & Closing
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()
