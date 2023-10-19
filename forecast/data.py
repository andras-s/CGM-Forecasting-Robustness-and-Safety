import os
import numpy as np
import pandas as pd
import torch
import gpytorch
import pickle
import json
import time
from tqdm import tqdm
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from preprocess.impute.data import GaussianProcessDataset
from preprocess.impute.inference import gp_sampling
from preprocess.impute.model import GaussianProcess
from utils.calc import get_zone
from utils.setup import ROOT_DIR, DATA_DIR
from utils.data.load import read_data

from utils.data.manipulate import create_windows, split_by, generate_predefined_split
import sys

sys.path.append(os.path.expanduser("~/Thesis/icarus/"))  # append repo location to $PYTHONPATH such that imports work


def get_data_loaders(train_data, val_data, test_data, args):
    if not args.pytorch_forecasting:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        return train_loader, val_loader, test_loader, None, None, None
    elif args.pytorch_forecasting:
        train_dataset, val_dataset, test_dataset = create_timeseries_datasets(train_data, val_data, test_data, args.features)
        train_loader = train_dataset.to_dataloader(train=True, batch_size=args.batch_size)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=len(val_data))
        test_loader = test_dataset.to_dataloader(train=False, batch_size=len(test_data))
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


class ForecastingDataset(GaussianProcessDataset):
    def __init__(self,
                 data,
                 file_ids,
                 load_dir,
                 mtype,
                 model,
                 features=None,
                 window_length=24,
                 sample_step=5,
                 window_start=6,
                 pred_start=22,
                 input_duration=2,
                 target_duration=1,
                 return_var=False,
                 resampling_method=None,
                 cut_input_target_at_measurement=True):
        """
        Creates a forecasting dataset from windowed data

        Parameters
            window_length   (int) the length (h) with which we sample from the GP
            sample_step     (int) the interval (min) with which we sample from the GP
            window_start    (int) the start in local time (h) of the sampled time window
            pred_start      (int) the start in local time (h) of the prediction
            input_duration  (int) the length (h) of the input of the RNN
            target_duration (int) the length (h) of the target/prediction (aka horizon)
        """
        super(ForecastingDataset, self).__init__(data)  # we first obtain the data set from the GP
        self.file_ids = file_ids
        self.load_dir = load_dir
        self.files_info = pd.read_excel(DATA_DIR + load_dir + 'files_info.xlsx')
        self.window_start = window_start
        self.mtype = mtype
        self.model = model
        if features is None:
            features = ['time']
        self.features = features
        self.window_length = window_length
        self.sample_step = sample_step
        self.window_start = window_start
        self.pred_start = pred_start
        self.input_duration = input_duration
        self.target_duration = target_duration
        self.return_var = return_var
        if resampling_method is None:
            resampling_method = ['linear', 'linear']
        self.resampling_method = resampling_method
        self.cut_input_target_at_measurement = cut_input_target_at_measurement

        self.num_samples = 0
        self.input_len = 0
        self.target_len = 0
        self.sample_len = 0
        self.window_data = None
        self.imputation_slicing_times = None

        self.input_means, self.input_vars, self.target_values = self.resample_glucose_values(data)
        self.model_data_df, self.feature_names = self.create_model_data_df()
        model_input = self.model_data_df.loc[self.model_data_df.time_idx < self.input_len, self.feature_names]
        model_input = torch.tensor(model_input.values)
        model_input = torch.reshape(model_input, (self.num_samples, self.input_len, len(self.feature_names)))

        # return inputs and targets in the correct shape (N, L, D_in)
        if mtype == 'regression':
            self.inputs = model_input
            self.target = torch.unsqueeze(self.target_values, -1)
        # maybe outdated:
        elif mtype == 'classification':
            print("Get classification labels ..")
            self.inputs = model_input
            self.target = torch.unsqueeze(torch.tensor([get_zone(y, zone='hypo', sample_step=sample_step) for y in tqdm(self.target_values)]), -1).int()

    def resample_glucose_values(self, data):    # todo: something goes wrong: libre is plotted too long and medtronic to short... (see notes)
        start_time_thresholds = []
        actual_cut_times = []
        end_time_thresholds = []
        first_times_resampled_input = []
        last_times_resampled_input = []
        first_times_resampled_target = []
        last_times_resampled_target = []
        first_times_used_for_input_imputation = []
        last_times_used_for_input_imputation = []
        first_times_used_for_target_imputation = []
        last_times_used_for_target_imputation = []
        first_indices_used_for_input_imputation = []
        last_indices_used_for_input_imputation = []
        first_indices_used_for_target_imputation = []
        last_indices_used_for_target_imputation = []
        cut_time_shifts = []
        empty_targets = []

        self.slice(self.sample_step, self.window_start, self.pred_start, self.input_duration, self.target_duration)
        num_windows = 0
        num_empty_targets = 0
        num_cut_shifts = 0
        input_means, input_vars, target_values = [], [], []
        for time, g_value, lens in zip(self.inputs, self.target, self.lengths):
            if self.cut_input_target_at_measurement:
                first_measurements_time_after_or_at_cut = next(t for t in time[:lens] if t >= self.cut_time)
                cut_time_shift = first_measurements_time_after_or_at_cut - self.cut_time
                if cut_time_shift > 0:
                    num_cut_shifts += 1
                start_time = self.start_time + cut_time_shift
                cut_time = self.cut_time + cut_time_shift
                end_time = self.end_time + cut_time_shift
            else:
                cut_time_shift = 0
                start_time = self.start_time
                cut_time = self.cut_time
                end_time = self.end_time

            start_time_threshold = self.start_time
            actual_cut_time = cut_time
            first_measurements_time_after_or_at_end = next(t for t in time[:lens] if t >= end_time)
            end_time_threshold = first_measurements_time_after_or_at_end

            resampled_input_times = torch.arange(start=start_time, end=actual_cut_time + self.sample_step,
                                                 step=self.sample_step)
            resampled_target_times = torch.arange(start=actual_cut_time, end=end_time + self.sample_step,
                                                  step=self.sample_step)

            first_time_used_for_input_imputation = next(
                time[i] for i in range(time[:lens].size()[0]) if time[i] >= start_time_threshold)
            last_time_used_for_input_imputation = next(
                time[i - 1] for i in range(time[:lens].size()[0]) if time[i] > actual_cut_time)
            first_time_used_for_target_imputation = next(
                time[i] for i in range(time[:lens].size()[0]) if time[i + 1] >= resampled_target_times[0])
            last_time_used_for_target_imputation = next(
                time[i] for i in range(time[:lens].size()[0]) if time[i] > resampled_target_times[-1])

            input_imputation_first_index = next(
                index for index, t in enumerate(time[:lens]) if t == first_time_used_for_input_imputation)
            input_imputation_last_index = next(
                index for index, t in enumerate(time[:lens]) if t == last_time_used_for_input_imputation)
            target_imputation_first_index = next(
                index for index, t in enumerate(time[:lens]) if t == first_time_used_for_target_imputation)
            target_imputation_last_index = next(
                index for index, t in enumerate(time[:lens]) if t == last_time_used_for_target_imputation)

            input_interpolation_times = time[input_imputation_first_index:input_imputation_last_index + 1]
            input_interpolation_values = g_value[input_imputation_first_index:input_imputation_last_index + 1]
            target_interpolation_times = time[target_imputation_first_index:target_imputation_last_index + 1]
            target_interpolation_values = g_value[target_imputation_first_index:target_imputation_last_index + 1]

            empty_target = True if len(target_interpolation_times) <= 1 else False
            num_empty_targets = num_empty_targets + 1 if len(
                target_interpolation_times) <= 1 else num_empty_targets  # Sometimes (especially with short horizons) missing values, cause the target array to be empty
            num_windows += 1

            # Get evenly spaced samples as input for the RNN by interpolating the measurements
            if self.resampling_method[0] == 'linear':
                model_input_means = torch.from_numpy(
                    np.interp(resampled_input_times, input_interpolation_times, input_interpolation_values))
                model_input_vars = torch.zeros_like(model_input_means)
            elif self.resampling_method[0] == 'gp':
                model_input_means, model_input_vars = gp_sampling(input_interpolation_times, input_interpolation_values,
                                                                  self.model, inputs_gp=resampled_input_times)
            # Get evenly spaced samples as targets for the RNN by interpolating the measurements
            if self.resampling_method[1] == 'linear':
                model_target_values = torch.from_numpy(
                    np.interp(resampled_target_times, target_interpolation_times, target_interpolation_values))
            elif self.resampling_method[1] == 'gp':
                model_target_values, _ = gp_sampling(target_interpolation_times, target_interpolation_values,
                                                     self.model,
                                                     inputs_gp=resampled_target_times)

            input_means.append(model_input_means)
            input_vars.append(model_input_vars)
            target_values.append(model_target_values)

            start_time_thresholds.append(start_time_threshold)
            actual_cut_times.append(actual_cut_time.item())
            end_time_thresholds.append(end_time_threshold.item())
            first_times_resampled_input.append(resampled_input_times[0].item())
            last_times_resampled_input.append(resampled_input_times[-1].item())
            first_times_resampled_target.append(resampled_target_times[0].item())
            last_times_resampled_target.append(resampled_target_times[-1].item())
            first_times_used_for_input_imputation.append(first_time_used_for_input_imputation.item())
            last_times_used_for_input_imputation.append(last_time_used_for_input_imputation.item())
            first_times_used_for_target_imputation.append(first_time_used_for_target_imputation.item())
            last_times_used_for_target_imputation.append(last_time_used_for_target_imputation.item())
            first_indices_used_for_input_imputation.append(input_imputation_first_index)
            last_indices_used_for_input_imputation.append(input_imputation_last_index)
            first_indices_used_for_target_imputation.append(target_imputation_first_index)
            last_indices_used_for_target_imputation.append(target_imputation_last_index)
            cut_time_shifts.append(cut_time_shift.item())
            empty_targets.append(empty_target)

        self.store_data_as_attributes(self.file_ids, data, cut_time_shifts, empty_targets,
                                      first_times_resampled_input, last_times_resampled_input,
                                      first_times_resampled_target, last_times_resampled_target,
                                      first_times_used_for_input_imputation, last_times_used_for_input_imputation,
                                      first_times_used_for_target_imputation, last_times_used_for_target_imputation,
                                      first_indices_used_for_input_imputation, last_indices_used_for_input_imputation,
                                      first_indices_used_for_target_imputation, last_indices_used_for_target_imputation)

        input_means = torch.stack(input_means).float()
        input_vars = torch.stack(input_vars).float()
        target_values = torch.stack(target_values).float()

        return input_means, input_vars, target_values

    def slice(self, sample_step, window_start, pred_start, input_duration, target_duration):
        """
        Calculates the input and target slice for time series forecasting
        First calculate the cut (h) between input and target. 
        The input window is sliced from cut - input_duration to the cut.
        The target window is sliced from cut to cut + target_duration.

        Parameters
            sample_step     (int) the interval (min) with which we sampled from the GP
            window_start    (int) the start in local time (h) of the sampled time windows
            pred_start      (int) the start in local time (h) of the prediction
            input_duration  (int) the duration of the slice meant as input of the RNN
            target_duration (int) the duration of the slice meant as target of the RNN (aka horizon)
        """
        if not pred_start:  # no pred start defined, just take the input length
            cut = input_duration
        else:  # if pred_start defined
            if window_start:
                cut = pred_start - window_start
            else:  # if pred_start is defined but no window_start (cannot happen)
                raise NotImplementedError("If you want predictions to start at a certain hour,\
                    please make sure your windows are starting from a certain hour \
                    (i.e. define window_start in both the impute and forecast config files. \
                    Else, please set pred_start to None and try again.")

        freq = int(60 / sample_step)
        self.inputs_slice = slice(freq * (cut - input_duration), freq * cut)
        self.target_slice = slice(freq * cut, int(freq * (cut + target_duration)))

        self.start_time = 60 * (cut - input_duration)
        self.cut_time = 60 * cut
        self.end_time = 60 * (cut + target_duration)

    def __len__(self):
        return self.input_means.shape[0]

    def __getitem__(self, i):
        return self.inputs[i], self.target[i]

    def store_data_as_attributes(self, file_ids, data, cut_time_shifts, empty_targets, first_times_resampled_input,
                                 last_times_resampled_input, first_times_resampled_target, last_times_resampled_target,
                                 first_times_used_for_input_imputation, last_times_used_for_input_imputation,
                                 first_times_used_for_target_imputation, last_times_used_for_target_imputation,
                                 first_indices_used_for_input_imputation, last_indices_used_for_input_imputation,
                                 first_indices_used_for_target_imputation, last_indices_used_for_target_imputation):
        # Save the corresponding window data to each entry to be able to reconstruct the original data
        self.window_data = pd.DataFrame()
        self.window_data['window_id'] = [f'{file_ids[i]}_{data[i].datetime[0]}' for i in range(len(file_ids))]
        self.window_data['file_id'] = [int(f_id) for f_id in file_ids]

        self.window_data = pd.merge(self.window_data, self.files_info, on='file_id', how='left')
        self.window_data['window_start_datetime'] = [data[i].datetime[0] for i in range(len(file_ids))]
        window_times = pd.DataFrame(data=self.inputs.numpy(), columns=[f'time_{i}' for i in range(self.inputs.size()[-1])])
        window_values = pd.DataFrame(data=self.target.numpy(), columns=[f'glucose_value_{i}' for i in range(self.target.size()[-1])])
        self.window_data = pd.concat([self.window_data, window_times, window_values], axis=1)

        self.imputation_slicing_times = pd.DataFrame()
        self.imputation_slicing_times['window_id'] = [f'{file_ids[i]}_{data[i].datetime[0]}' for i in range(len(file_ids))]
        self.imputation_slicing_times['cut_shift'] = cut_time_shifts
        self.imputation_slicing_times['empty_target'] = empty_targets
        self.imputation_slicing_times['first_time_resampled_input'] = first_times_resampled_input
        self.imputation_slicing_times['last_time_resampled_input'] = last_times_resampled_input
        self.imputation_slicing_times['first_time_resampled_target'] = first_times_resampled_target
        self.imputation_slicing_times['last_time_resampled_target'] = last_times_resampled_target
        self.imputation_slicing_times['first_dt_resampled_input'] = self.window_data['window_start_datetime'] + pd.to_timedelta(first_times_resampled_input, unit='min')
        self.imputation_slicing_times['last_dt_resampled_input'] = self.window_data['window_start_datetime'] + pd.to_timedelta(last_times_resampled_input, unit='min')
        self.imputation_slicing_times['first_dt_resampled_target'] = self.window_data['window_start_datetime'] + pd.to_timedelta(first_times_resampled_target, unit='min')
        self.imputation_slicing_times['last_dt_resampled_target'] = self.window_data['window_start_datetime'] + pd.to_timedelta(last_times_resampled_target, unit='min')
        self.imputation_slicing_times['first_time_used_for_input_imputation'] = first_times_used_for_input_imputation
        self.imputation_slicing_times['last_time_used_for_input_imputation'] = last_times_used_for_input_imputation
        self.imputation_slicing_times['first_time_used_for_target_imputation'] = first_times_used_for_target_imputation
        self.imputation_slicing_times['last_time_used_for_target_imputation'] = last_times_used_for_target_imputation
        self.imputation_slicing_times['first_index_used_for_input_imputation'] = first_indices_used_for_input_imputation
        self.imputation_slicing_times['last_index_used_for_input_imputation'] = last_indices_used_for_input_imputation
        self.imputation_slicing_times['first_index_used_for_target_imputation'] = first_indices_used_for_target_imputation
        self.imputation_slicing_times['last_index_used_for_target_imputation'] = last_indices_used_for_target_imputation

    def create_model_data_df(self):
        self.num_samples = self.input_means.size(0)
        self.input_len = self.input_means.size(1)
        self.target_len = self.target_values.size(1)
        self.sample_len = self.input_len + self.target_len

        first_input_dt = np.array(self.imputation_slicing_times['first_dt_resampled_input'])
        input_dts = np.array([first_input_dt + np.timedelta64(i * self.sample_step, 'm') for i in range(self.input_len)]).transpose()
        first_target_dt = np.array(self.imputation_slicing_times['first_dt_resampled_target'])
        target_dts = np.array([first_target_dt + np.timedelta64(i * self.sample_step, 'm') for i in range(self.target_len)]).transpose()
        dts_array = np.concatenate((input_dts, target_dts), axis=1)
        dts = pd.to_datetime(dts_array.flatten().tolist())

        glucose_value = torch.cat((self.input_means, self.target_values), axis=1).flatten().tolist()
        window_ids = self.window_data.window_id
        window_ids_unique = window_ids.unique()
        window_ids_stacked = sum([[win_id] * self.sample_len for win_id in window_ids_unique], [])
        id_dict = dict(zip(window_ids_unique, range(len(window_ids_unique))))
        window_num = window_ids.replace(id_dict)
        window_num_stacked = sum([[win_id] * self.sample_len for win_id in window_num], [])
        time_idx_stacked = [time_idx for time_idx in range(self.sample_len)] * self.num_samples

        column_names = ['window_num', 'window_id', 'time_idx', 'datetime', 'glucose_value']
        feature_names = ['glucose_value']
        columns = (window_num_stacked, window_ids_stacked, time_idx_stacked, dts, glucose_value)
        if 'time' in self.features:
            mins_of_day = (dts_array.astype('datetime64[m]') - dts_array.astype('datetime64[D]')).astype(int)
            mins_of_day_scaled = mins_of_day * np.pi / 720  # * 2pi/(60*24)
            time_of_day_x = np.cos(mins_of_day_scaled)
            time_of_day_y = np.sin(mins_of_day_scaled)
            time_of_day_x = time_of_day_x.flatten().tolist()
            time_of_day_y = time_of_day_y.flatten().tolist()
            column_names = column_names + ['time_x', 'time_y']
            feature_names = feature_names + ['time_x', 'time_y']
            columns = (*columns, time_of_day_x, time_of_day_y)
        if 'day' in self.features:
            day = pd.Series(dts_array.flatten()).dt.dayofweek.tolist()
            column_names = column_names + ['day']
            feature_names = feature_names + ['day']
            columns = (*columns, day)
        if 'glucose_delta' in self.features:
            glucose_delta = np.array(glucose_value) - np.roll(glucose_value, 1)
            glucose_delta[0::self.sample_len] = [0] * self.num_samples
            glucose_delta = glucose_delta.tolist()
            column_names = column_names + ['glucose_delta']
            feature_names = feature_names + ['glucose_delta']
            columns = (*columns, glucose_delta)
        if 'diff_10' in self.features:
            diff_10 = np.array(glucose_value) - np.roll(glucose_value, 2)
            diff_10[0::self.sample_len] = [0] * self.num_samples
            diff_10[1::self.sample_len] = [1] * self.num_samples
            diff_10 = diff_10.tolist()
            column_names = column_names + ['diff_10']
            feature_names = feature_names + ['diff_10']
            columns = (*columns, diff_10)
        if 'short_avg_delta' in self.features:
            glucose_roc_1 = (np.array(glucose_value) - np.roll(glucose_value, 1)) / 5
            glucose_roc_2 = (np.array(glucose_value) - np.roll(glucose_value, 2)) / 10
            glucose_roc_3 = (np.array(glucose_value) - np.roll(glucose_value, 3)) / 15
            short_avg_delta = (glucose_roc_1 + glucose_roc_2 + glucose_roc_3) / 3
            short_avg_delta[0::self.sample_len] = [0] * self.num_samples
            short_avg_delta[1::self.sample_len] = [0] * self.num_samples
            short_avg_delta[2::self.sample_len] = [0] * self.num_samples
            short_avg_delta = short_avg_delta.tolist()
            column_names = column_names + ['short_avg_delta']
            feature_names = feature_names + ['short_avg_delta']
            columns = (*columns, short_avg_delta)
        if 'long_avg_delta' in self.features:
            glucose_roc = np.zeros((6, self.sample_len * self.num_samples))
            for num, i in enumerate([4, 5, 6, 7, 8, 9]):
                glucose_roc[num, :] = (np.array(glucose_value) - np.roll(glucose_value, i)) / (i*5)
                glucose_roc[num, (i-1)::self.sample_len] = [0] * self.num_samples
            long_avg_delta = np.sum(glucose_roc, axis=0) / 6
            long_avg_delta = long_avg_delta.tolist()
            column_names = column_names + ['long_avg_delta']
            feature_names = feature_names + ['long_avg_delta']
            columns = (*columns, long_avg_delta)
        if 'mean' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            mean_30 = glucose_value_series.rolling(6).mean().fillna(0)
            mean_30 = (mean_30 * (([0] * 6 + [1] * (self.sample_len-6)) * self.num_samples)).tolist()
            mean_60 = glucose_value_series.rolling(12).mean().fillna(0)
            mean_60 = (mean_60 * (([0] * 12 + [1] * (self.sample_len-12)) * self.num_samples)).tolist()
            mean_120 = glucose_value_series.rolling(24).mean().fillna(0)
            mean_120 = (mean_120 * (([0] * 24 + [1] * (self.sample_len-24)) * self.num_samples)).tolist()
            column_names = column_names + ['mean_30', 'mean_60', 'mean_120']
            feature_names = feature_names + ['mean_30', 'mean_60', 'mean_120']
            columns = (*columns, mean_30, mean_60, mean_120)
        if 'std' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            std_30 = glucose_value_series.rolling(6).std().fillna(0)
            std_30 = (std_30 * (([0] * 6 + [1] * (self.sample_len-6)) * self.num_samples)).tolist()
            std_60 = glucose_value_series.rolling(12).std().fillna(0)
            std_60 = (std_60 * (([0] * 12 + [1] * (self.sample_len-12)) * self.num_samples)).tolist()
            std_120 = glucose_value_series.rolling(24).std().fillna(0)
            std_120 = (std_120 * (([0] * 24 + [1] * (self.sample_len-24)) * self.num_samples)).tolist()
            column_names = column_names + ['std_30', 'std_60', 'std_120']
            feature_names = feature_names + ['std_30', 'std_60', 'std_120']
            columns = (*columns, std_30, std_60, std_120)
        if 'coefficient_variation' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            cv_30 = (glucose_value_series.rolling(6).std() / glucose_value_series.rolling(6).mean()).fillna(0)
            cv_30 = (cv_30 * (([0] * 6 + [1] * (self.sample_len-6)) * self.num_samples)).tolist()
            cv_60 = (glucose_value_series.rolling(12).std() / glucose_value_series.rolling(12).mean()).fillna(0)
            cv_60 = (cv_60 * (([0] * 12 + [1] * (self.sample_len-12)) * self.num_samples)).tolist()
            cv_120 = (glucose_value_series.rolling(24).std() / glucose_value_series.rolling(24).mean()).fillna(0)
            cv_120 = (cv_120 * (([0] * 24 + [1] * (self.sample_len-24)) * self.num_samples)).tolist()
            column_names = column_names + ['cv_30', 'cv_60', 'cv_120']
            feature_names = feature_names + ['cv_30', 'cv_60', 'cv_120']
            columns = (*columns, cv_30, cv_60, cv_120)
        if 'pos_increments_2h' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            glucose_delta = glucose_value_series.diff().fillna(0)
            glucose_delta_pos = glucose_delta
            glucose_delta_pos[glucose_delta < 0] = 0
            pos_increments_2h = glucose_delta_pos.rolling(24).sum().fillna(0)
            pos_increments_2h = (pos_increments_2h * (([0] * 24 + [1] * (self.sample_len - 24)) * self.num_samples)).tolist()
            column_names = column_names + ['pos_increments_2h']
            feature_names = feature_names + ['pos_increments_2h']
            columns = (*columns, pos_increments_2h)
        if 'neg_increments_2h' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            glucose_delta = glucose_value_series.diff().fillna(0)
            glucose_delta_pos = glucose_delta
            glucose_delta_pos[glucose_delta > 0] = 0
            neg_increments_2h = glucose_delta_pos.rolling(24).sum().fillna(0)
            neg_increments_2h = (neg_increments_2h * (([0] * 24 + [1] * (self.sample_len - 24)) * self.num_samples)).tolist()
            column_names = column_names + ['neg_increments_2h']
            feature_names = feature_names + ['neg_increments_2h']
            columns = (*columns, neg_increments_2h)
        if 'max_pos_increment_2h' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            glucose_delta = glucose_value_series.diff().fillna(0)
            glucose_delta_pos = glucose_delta
            glucose_delta_pos[glucose_delta < 0] = 0
            max_pos_increment_2h = glucose_delta_pos.rolling(24).max().fillna(0)
            max_pos_increment_2h = (max_pos_increment_2h * (([0] * 24 + [1] * (self.sample_len - 24)) * self.num_samples)).tolist()
            column_names = column_names + ['max_pos_increment_2h']
            feature_names = feature_names + ['max_pos_increment_2h']
            columns = (*columns, max_pos_increment_2h)
        if 'max_neg_increment_2h' in self.features:
            glucose_value_series = pd.Series(glucose_value)
            glucose_delta = glucose_value_series.diff().fillna(0)
            glucose_delta_pos = glucose_delta
            glucose_delta_pos[glucose_delta > 0] = 0
            max_neg_increment_2h = glucose_delta_pos.rolling(24).min().fillna(0)
            max_neg_increment_2h = (max_neg_increment_2h * (([0] * 24 + [1] * (self.sample_len - 24)) * self.num_samples)).tolist()
            column_names = column_names + ['max_neg_increment_2h']
            feature_names = feature_names + ['max_neg_increment_2h']
            columns = (*columns, max_neg_increment_2h)
        if self.return_var:
            target_vars = torch.zeros(target_dts.shape)
            variance = torch.cat((self.input_vars, target_vars), axis=1).flatten().tolist()
            column_names = column_names + ['variance']
            feature_names = feature_names + ['variance']
            columns = (*columns, variance)
        model_data_df = pd.DataFrame(list(zip(*columns)), columns=column_names)

        return model_data_df, feature_names


def check_imputed_data_exists(imputed_data_path, clean_data_path, args, gp_args):
    first_clean_file_name = os.listdir(clean_data_path)[0]
    first_clean_file_path = os.path.join(clean_data_path, first_clean_file_name)
    clean_data_timestamp = os.path.getmtime(first_clean_file_path)
    gp_file_path = os.path.join(ROOT_DIR, 'preprocess/impute/andras/results', args.gp_run_name)
    new_imputation_info = {
        'split_attribute': gp_args.split_attribute,
        'num_folds': gp_args.K,
        'window_length': gp_args.window_length,
        'max_gap': gp_args.max_gap,
        'tol_length': gp_args.tol_length,
        'window_start': gp_args.window_start,
        'input_duration': args.input_duration,
        'target_duration': args.target_duration,
        'pred_start': args.pred_start,
        'sample_step': args.sample_step,
        'resampling_method': args.resampling_method,
        'cut_input_target_at_measurement': args.cut_input_target_at_measurement,
        # 'clean_data_timestamp': clean_data_timestamp,
        'clean_data_timestamp': 1659447892.7317684,
        'gp_timestamp': os.path.getmtime(gp_file_path),
        'features': args.features
    }
    folds = [k for k in range(gp_args.K)]
    # Check whether imputed data already exists
    for imputed_data_dir in os.listdir(imputed_data_path):
        imputed_data_dir_path = os.path.join(imputed_data_path, imputed_data_dir)
        if os.path.isdir(imputed_data_dir_path):
            try:
                imputation_info = json.load(open(os.path.join(imputed_data_dir_path, 'imputation_info.json')))
                if imputation_info == new_imputation_info:
                    print('Data already exists')
                    return imputed_data_dir_path, folds, new_imputation_info
            except:
                print(f'{imputed_data_dir} does not contain an info file.')
    return None, folds, new_imputation_info


def impute_data(data_generator, imputed_data_path, clean_data_path, args, gp_args, device):
    print("Imputing the data ...")
    # load and resample data
    data_args = {**{k: v for k, v in gp_args.items() if k in ['window_start', 'window_length']},
                 **{k: v for k, v in args.items() if
                    k in ['features', 'sample_step', 'input_duration', 'target_duration', 'pred_start', 'resampling_method',
                          'cut_input_target_at_measurement']}}

    _, folds, imputation_info = check_imputed_data_exists(imputed_data_path, clean_data_path, args, gp_args)
    imputed_data_path = os.path.join(imputed_data_path, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(imputed_data_path)
    with open(os.path.join(imputed_data_path, 'imputation_info.json'), 'w') as f:
        json.dump(imputation_info, f, indent=4)

    for k, ((train_dfs, val_dfs, test_dfs), (train_file_ids, val_file_ids, test_file_ids)) in tqdm(enumerate(data_generator)):
        # load pretrained GP model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        if 'gp' in args.resampling_method:
            gp_model = GaussianProcess(None, None, likelihood, gp_args).to(device)
            gp_model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.getcwd()), 'preprocess/impute/andras/results', args.gp_run_name, f'models/weights_{1}.pth'), map_location=device))
            gp_model.device = device
        else:
            gp_model = None

        train_data = ForecastingDataset(data=train_dfs, file_ids=train_file_ids, load_dir=args.load_dir, mtype=args.mtype, model=gp_model, **data_args)
        val_data = ForecastingDataset(data=val_dfs, file_ids=val_file_ids, load_dir=args.load_dir, mtype=args.mtype, model=gp_model, **data_args)
        test_data = ForecastingDataset(data=test_dfs, file_ids=test_file_ids, load_dir=args.load_dir, mtype=args.mtype, model=gp_model, **data_args)

        with open(os.path.join(imputed_data_path, f'train_data_fold_{k}.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(imputed_data_path, f'val_data_fold_{k}.pkl'), 'wb') as f:
            pickle.dump(val_data, f)
        with open(os.path.join(imputed_data_path, f'test_data_fold_{k}.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

    return imputed_data_path, folds


def get_data_location(args, gp_args, device):
    clean_data_path = os.path.join(DATA_DIR, args.load_dir)
    imputed_data_path  = os.path.join(DATA_DIR, "4_imputed_data")
    if not os.path.exists(imputed_data_path):
        os.makedirs(imputed_data_path)

    # Check if data already exists. If it exists, get its path.
    data_path, folds, imputation_info = check_imputed_data_exists(imputed_data_path, clean_data_path, args, gp_args)
    if not args.force_new_imputation and data_path is not None:
        return data_path, folds

    # If data does not exist. Create windows and impute.
    data, file_ids = read_data(load_dir=args.load_dir)
    windows = create_windows(data, file_ids, window_length=gp_args.window_length, max_gap=gp_args.max_gap, tol_length=gp_args.tol_length, window_start=gp_args.window_start)
    data, file_ids, fold_ids = split_by(data=windows, attribute=gp_args.split_attribute, load_dir=args.load_dir, K=gp_args.K, seed=gp_args.seed)
    data_generator = generate_predefined_split(data, file_ids, fold_ids)
    data_path, folds = impute_data(list(data_generator), imputed_data_path, clean_data_path, args, gp_args, device)
    return data_path, folds


def load_data(k, data_path, args):
    print('Loading data ...')
    train_data = pd.read_pickle(os.path.join(data_path, f'train_data_fold_{k}.pkl'))
    val_data = pd.read_pickle(os.path.join(data_path, f'val_data_fold_{k}.pkl'))
    test_data = pd.read_pickle(os.path.join(data_path, f'test_data_fold_{k}.pkl'))
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_data_loaders(train_data, val_data, test_data, args)

    return train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_timeseries_df(data):
    num_samples = len(data.inputs)
    len_sample = len(data.inputs[0]) + len(data.target[0])

    samples = torch.cat([data.inputs, data.target], axis=1)
    samples_stacked = samples.flatten().tolist()
    window_ids = data.window_data.window_id
    window_ids_unique = window_ids.unique()
    id_dict = dict(zip(window_ids_unique, range(len(window_ids_unique))))
    window_num = window_ids.replace(id_dict)
    window_num_stacked = sum([[win_id] * len_sample for win_id in window_num], [])
    time_idx_stacked = [time_idx for time_idx in range(len_sample)] * num_samples
    timeseries_fold_df = pd.DataFrame(list(zip(samples_stacked, window_num_stacked, time_idx_stacked)),
                                      columns=['glucose_value', 'window_id', 'time_idx'])
    return timeseries_fold_df


def create_timeseries_datasets(train_data, val_data, test_data, features):
    print('Creating time series datasets ...')
    train_timeseries_df = train_data.model_data_df
    val_timeseries_df = val_data.model_data_df
    test_timeseries_df = test_data.model_data_df

    static_categoricals = []
    if 'day' in features:
        static_categoricals = static_categoricals + ['day']

    time_varying_known_reals = ["time_idx"]
    if 'time' in features:
        time_varying_known_reals = time_varying_known_reals + ['time_x', 'time_y']

    time_varying_unknown_reals = ["glucose_value"]
    if 'glucose_delta' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['glucose_delta']
    if 'diff_10' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['diff_10']
    if 'short_avg_delta' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['short_avg_delta']
    if 'long_avg_delta' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['long_avg_delta']
    if 'mean' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['mean_30', 'mean_60', 'mean_120']
    if 'std' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['std_30', 'std_60', 'std_120']
    if 'coefficient_variation' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['cv_30', 'cv_60', 'cv_120']
    if 'pos_increments_2h' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['pos_increments_2h']
    if 'neg_increments_2h' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['neg_increments_2h']
    if 'max_pos_increment_2h' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['max_pos_increment_2h']
    if 'max_neg_increment_2h' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['max_neg_increment_2h']
    if 'glucose_delta' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['glucose_delta']
    if 'glucose_delta' in features:
        time_varying_unknown_reals = time_varying_unknown_reals + ['glucose_delta']

    train_timeseries_dataset = TimeSeriesDataSet(train_timeseries_df,
                                                 time_idx="time_idx",
                                                 target="glucose_value",
                                                 group_ids=["window_num"],
                                                 min_encoder_length=train_data.input_len,
                                                 max_encoder_length=train_data.input_len,
                                                 min_prediction_length=train_data.target_len,
                                                 max_prediction_length=train_data.target_len,
                                                 static_categoricals=static_categoricals,
                                                 # static_reals=[],
                                                 # time_varying_known_categoricals=[],
                                                 # variable_groups={},
                                                 time_varying_known_reals=time_varying_known_reals,   # Very important!!!
                                                 # time_varying_unknown_categoricals=[],
                                                 time_varying_unknown_reals=time_varying_unknown_reals,
                                                 target_normalizer=None,
                                                 add_relative_time_idx=True,
                                                 add_target_scales=False,
                                                 add_encoder_length=True,
                                                 # scalers={'glucose_value': EncoderNormalizer(method='identity')}
                                                 )
    val_timeseries_dataset = TimeSeriesDataSet.from_dataset(train_timeseries_dataset, val_timeseries_df, predict=True, stop_randomization=True)
    test_timeseries_dataset = TimeSeriesDataSet.from_dataset(train_timeseries_dataset, test_timeseries_df, predict=True, stop_randomization=True)
    return train_timeseries_dataset, val_timeseries_dataset, test_timeseries_dataset

    # print('Creating time series datasets ...')
    # len_input = len(train_data.inputs[0])
    # len_target = len(train_data.target[0])
    # train_timeseries_df = create_timeseries_df(train_data)
    # val_timeseries_df = create_timeseries_df(val_data)
    # train_timeseries_dataset = TimeSeriesDataSet(train_timeseries_df,
    #                                              time_idx="time_idx",
    #                                              target="glucose_value",
    #                                              group_ids=["window_id"],
    #                                              min_encoder_length=len_input,
    #                                              max_encoder_length=len_input,
    #                                              min_prediction_length=len_target,
    #                                              max_prediction_length=len_target,
    #                                              #static_categoricals=["window_id"],
    #                                              #static_reals=[],
    #                                              #time_varying_known_categoricals=[],
    #                                              #variable_groups={},
    #                                              time_varying_known_reals=["time_idx"],   # Very important!!!
    #                                              #time_varying_unknown_categoricals=[],
    #                                              time_varying_unknown_reals=["glucose_value"],
    #                                              target_normalizer=None,
    #                                              add_relative_time_idx=True,
    #                                              add_target_scales=False,
    #                                              add_encoder_length=True,
    #                                              # scalers={'glucose_value': EncoderNormalizer(method='identity')}
    #                                              )
    # val_timeseries_dataset = TimeSeriesDataSet.from_dataset(train_timeseries_dataset, val_timeseries_df, predict=True, stop_randomization=True)
    # return train_timeseries_dataset, val_timeseries_dataset
