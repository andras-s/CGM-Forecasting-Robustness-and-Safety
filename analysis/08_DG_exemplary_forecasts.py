from utils.plot import compare_forecasts
from analysis.helper import ModelLoaderEvaluator, get_windows_with_largest_discrepancy, get_forecast_data

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", message="Negative variance values detected. This is likely due to numerical instabilities. Rounding negative variances up to 1e-06.")
pd.set_option('display.float_format', lambda x: '%.9f' % x)
np.set_printoptions(suppress=True)

path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment'
seed = 0
fold = 2
horizon = 1
base_model_type = 'single'
advanced_model_type = 'E2E_MMD'
gpu = '0'
num_windows = 10
domain_name = 'test'


base_model_evaluator = ModelLoaderEvaluator(path, base_model_type, seed, horizon, fold, gpu)
advanced_model_evaluator = ModelLoaderEvaluator(path, advanced_model_type, seed, horizon, fold, gpu)

windows_of_interest_nll, windows_of_interest_peg = get_windows_with_largest_discrepancy(
    model_evaluation_0=base_model_evaluator.evaluation,
    model_evaluation_1=advanced_model_evaluator.evaluation,
    criterion='NLL',
    num_windows=num_windows
)

for win_id in windows_of_interest_nll.window_id:
    window_nll = windows_of_interest_nll[windows_of_interest_nll.window_id == win_id]
    file_id, date, input_datetimes, inputs, target_datetimes, targets, mus_nll, sigmas_nll, measure_datetimes, measure_values, kpis_nll = get_forecast_data(sample=window_nll, model_evaluation=base_model_evaluator.evaluation)

    window_peg = windows_of_interest_peg[windows_of_interest_peg.window_id == win_id]
    _, _, _, _, _, _, mus_peg, sigmas_peg, _, _, kpis_peg = get_forecast_data(sample=window_peg, model_evaluation=advanced_model_evaluator.evaluation)

    forecast_plots_path = os.path.join(base_model_evaluator.evaluation.args.base_dir, base_model_evaluator.evaluation.args.mtype, base_model_evaluator.evaluation.args.save_string, "plots", "forecast")

    base_model_name = base_model_type
    advanced_model_name = advanced_model_type if '_' not in advanced_model_type else advanced_model_type.replace('_', ' ')

    compare_forecasts(
        input_datetimes,
        inputs,
        target_datetimes,
        targets,
        mus_nll,
        mus_peg,
        sigmas_nll,
        sigmas_peg,
        model_0_name = base_model_name,
        model_1_name = advanced_model_name,
        save_to=f"{forecast_plots_path}/compare_forecast_{file_id}",
        plot_show=True
    )

    kpis_base_string     = f'{base_model_name}:   ' + ''.join([f"{k}: {round(v, 2)}   " for k, v in kpis_nll.items()]) if kpis_nll else ''
    kpis_advanced_string = f'{advanced_model_name}:   ' + ''.join([f"{k}: {round(v, 2)}   " for k, v in kpis_peg.items()]) if kpis_peg else ''

    print(f"\n-----   patient {window_nll.patient_id_study.item()} on {window_nll.window_start_datetime.item()}   -----")
    print(kpis_base_string, '\n', kpis_advanced_string)
