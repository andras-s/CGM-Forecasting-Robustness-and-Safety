from forecast.data import load_data
from forecast.helper import get_metrics, get_model
from forecast.evaluation import ModelEvaluator, AggregateEvaluator
from utils.setup import init_gpu
from utils.utils import dotdict
from utils.plot import compare_forecasts
from analysis.helper import get_windows_with_largest_discrepancy, get_forecast_data

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

# horizon = 0.5
run_path_nll = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/LSTM_NLL_i6_hor0.5'
run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/LSTM_NLLPEGSurface_i6_hor0.5'

# horizon = 1
# run_path_nll = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/LSTM_NLL_i6_hor1'
# run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/LSTM_NLLPEGSurface_i6_hor1'

# horizon = 2
# run_path_nll = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/LSTM_NLL_i6_hor2'
# run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/LSTM_NLLPEGSurface_i6_hor2'

dg = False
domain_name = 'test'
num_windows = 10
gpu = '0'

run_paths = [run_path_nll, run_path_peg]
model_evaluations = []
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
    args.plot_show = False
    if not os.path.exists(args.save_string):
        os.makedirs(args.save_string)

    # with open(os.path.join(run_path, 'logs.pkl'), 'rb') as f:
    #     logs = pickle.load(f)
    logs = None

    k = folds[2]
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
    model_evaluations.append(model_evaluation)


windows_of_interest_nll, windows_of_interest_peg = get_windows_with_largest_discrepancy(
    model_evaluation_0=model_evaluations[0],
    model_evaluation_1=model_evaluations[1],
    num_windows=num_windows
)

for win_id in windows_of_interest_nll.window_id:
    window_nll = windows_of_interest_nll[windows_of_interest_nll.window_id == win_id]
    file_id, date, input_datetimes, inputs, target_datetimes, targets, mus_nll, sigmas_nll, measure_datetimes, measure_values, kpis_nll = get_forecast_data(sample=window_nll, model_evaluation=model_evaluations[0])

    window_peg = windows_of_interest_peg[windows_of_interest_peg.window_id == win_id]
    _, _, _, _, _, _, mus_peg, sigmas_peg, _, _, kpis_peg = get_forecast_data(sample=window_peg, model_evaluation=model_evaluations[1])

    forecast_plots_path = os.path.join(model_evaluations[0].args.base_dir, model_evaluations[0].args.mtype, model_evaluations[0].args.save_string, "plots", "forecast")
    if not os.path.exists(forecast_plots_path):
        os.makedirs(forecast_plots_path)

    compare_forecasts(
        input_datetimes,
        inputs,
        target_datetimes,
        targets,
        mus_nll,
        mus_peg,
        sigmas_nll,
        sigmas_peg,
        save_to=f"{forecast_plots_path}/compare_forecast_{file_id}",
        plot_show=True
    )

    kpis_nll_string = 'NLL:   ' + ''.join([f"{k}: {round(v, 2)}   " for k, v in kpis_nll.items()]) if kpis_nll else ''
    kpis_peg_string = 'PEG:   ' + ''.join([f"{k}: {round(v, 2)}   " for k, v in kpis_peg.items()]) if kpis_peg else ''

    print(f"\n-----   patient {window_nll.patient_id_study.item()} on {window_nll.window_start_datetime.item()}   -----")
    print(kpis_nll_string, '\n', kpis_peg_string)
