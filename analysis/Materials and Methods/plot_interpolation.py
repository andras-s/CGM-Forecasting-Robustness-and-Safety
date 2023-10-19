import os
import datetime
import random
import numpy as np
import torch
from tqdm import tqdm
from forecast.data import load_data
from forecast.helper import initialize_run
from utils.plot import plot_interpolation
from utils.setup import ROOT_DIR

random.seed(0)

# INPUT
num_plots = 5
fold = 0
save_dir = os.path.join(ROOT_DIR, 'analysis', 'results/interpolation')

# MAIN
data_path, _, _, args, _ = initialize_run()
train_data, _, _, _, _, _, _, _, _ = load_data(fold, data_path, args)

input_data = torch.squeeze(train_data.inputs)
target_data = torch.squeeze(train_data.target)
window_data = train_data.window_data
interp_data = train_data.imputation_slicing_times
sample_step = train_data.sample_step

print('Plotting ...')
indices_plot = random.sample(list(window_data.index), num_plots)
for index in tqdm(indices_plot):
    window_entry = window_data.loc[index]
    interp_entry = interp_data.loc[index]
    window_start_dt = window_entry.window_start_datetime

    # measurement data
    first_index = interp_entry.first_index_used_for_input_imputation.item()
    last_index = interp_entry.last_index_used_for_target_imputation.item()
    measure_times = [window_entry[f'time_{ind}'].item() for ind in range(first_index, last_index)]
    measure_dts = [window_start_dt + datetime.timedelta(minutes=int(rel_time)) for rel_time in measure_times]
    measure_values = [window_entry[f'glucose_value_{ind}'].item() for ind in range(first_index, last_index)]

    # input data
    first_input_time = interp_entry.first_time_resampled_input.item()
    last_input_time = interp_entry.last_time_resampled_input.item()
    input_times = np.arange(first_input_time, last_input_time + 1, step=sample_step, dtype=int)
    input_dts = [window_start_dt + datetime.timedelta(minutes=int(rel_time)) for rel_time in input_times]
    input_values = input_data[index, :]

    # target data
    first_target_time = interp_entry.first_time_resampled_target.item()
    last_target_time = interp_entry.last_time_resampled_target.item()
    target_times = np.arange(first_target_time, last_target_time + 1, step=sample_step, dtype=int)
    target_dts = [window_start_dt + datetime.timedelta(minutes=int(rel_time)) for rel_time in target_times][1:]
    target_values = target_data[index, 1:]

    # plot
    plot_interpolation(
        measure_datetimes=measure_dts,
        measure_values=measure_values,
        input_datetimes=input_dts,
        inputs=input_values,
        target_datetimes=target_dts,
        targets=target_values,
        window_id=window_entry.window_id,
        save_dir=save_dir
    )

temp = 0
