from forecast.data import load_data
from forecast.helper import get_metrics, get_model
from forecast.evaluation import ModelEvaluator, AggregateEvaluator
from utils.setup import init_gpu
from utils.utils import dotdict

import pandas as pd
import numpy as np
import pickle
import json
import random
import sys
import os
import time
import torch
import warnings
warnings.filterwarnings("ignore", message="Negative variance values detected. This is likely due to numerical instabilities. Rounding negative variances up to 1e-06.")
pd.set_option('display.float_format', lambda x: '%.9f' % x)
np.set_printoptions(suppress=True)


# DG runs evaluations:
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/LSTM_single'
# experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/LSTM_ensemble'
# experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/LSTM_Layer/FT_MMD'
# experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/LSTM_Layer/FT_MMD'
dg = True
gpu = '1'
seeds = [0, 1, 2, 3, 4]
horizons = [0.5, 1, 2]
run_paths = []
seed_folders = os.listdir(experiment_path)
for h in horizons:
    for s in seeds:
        seed_folder_name = [folder for folder in seed_folders if f'_{s}_' in folder][0]
        seed_path = os.path.join(experiment_path, seed_folder_name)
        horizon_folders = os.listdir(seed_path)
        horizon_folder_name = [folder for folder in horizon_folders if f'_{s}_{h}_' in folder][0]
        horizon_path = os.path.join(seed_path, horizon_folder_name)
        run_paths.append(horizon_path)

# CV runs evaluation:
# dg = False
# runs_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment'
# run_paths = [os.path.join(runs_path, run_dir) for run_dir in os.listdir(runs_path)]

# single run evaluation:
# dg = False
# run_paths = ['/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/run_20220825-094114_ConvT_NLLPEGSurface_i6_hor0.5_b1024_lr0.001_g0.999_seed0_linearlinear']

for run_path in run_paths:
    print(f'\n Evaluating models at : {run_path}')
    # Initialize Evaluation with same args and data as original run. Adapt save string, but keep load string for model loading
    with open(run_path + '/args.json') as file:
        args = json.load(file)
        args = dotdict(args)
    with open(run_path + '/gp_args.json') as file:
        gp_args = json.load(file)
        gp_args = dotdict(gp_args)

    args.base_dir = '/local/home/ansass/Thesis/icarus/forecast/results/rnn'
    sys.path.append(os.path.expanduser(args.repo_location))
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = init_gpu(gpu)
    data_path = args.data_path
    folds = args.folds if dg else gp_args.folds

    args.save_string = run_path + '/evaluations/' + time.strftime('%Y%m%d-%H%M%S')
    args.domain = 'treatment'
    if not os.path.exists(args.save_string):
        os.makedirs(args.save_string)

    with open(os.path.join(run_path, 'logs.pkl'), 'rb') as f:
        logs = pickle.load(f)
    results = []
    for k in folds:
        print("\nData Split: ", k)
        # load data, model & metrics
        train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(k, data_path, args)
        model, optimiser, scheduler = get_model(args, device, train_data, train_dataset, num_input_features=train_data[0][0].size(1), input_length=train_data[0][0].size(0), output_size=train_data[0][-1].size(0), fold=k)
        model.load_state_dict(torch.load(os.path.join(run_path, "models", f"weights_{k}.pth")))
        del train_loader, val_loader, test_loader, optimiser, scheduler
        torch.cuda.empty_cache()
        model.eval()
        # evaluate and save
        criterion, metrics = get_metrics(args, train_data, device)
        model_evaluation = ModelEvaluator(model,
                                          train_data, val_data, test_data,
                                          train_dataset, val_dataset, test_dataset,
                                          criterion, metrics, args, k, logs)
        results.append(model_evaluation.results)
        del train_data, val_data, test_data, train_dataset, val_dataset, test_dataset, model, model_evaluation
        torch.cuda.empty_cache()
    AggregateEvaluator(results, logs, args)
