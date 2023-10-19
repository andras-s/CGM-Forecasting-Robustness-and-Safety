import os
import datetime
import random
import pandas as pd
import numpy as np
import torch

from forecast.helper import move_to
from utils.setup import DOMAINS
from utils.plot import plot_forecast, plot_error_grid, plot_pointwise_metric, visualize_model_performance
pd.options.display.float_format = '{:.4f}'.format


class ModelEvaluator:
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 test_data,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 criterion,
                 metrics,
                 args,
                 k,
                 logs,
                 skip_eval=False):
        self.model = model
        self.device = model.device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = criterion
        self.metrics = metrics
        self.args = args
        self.k = k
        self.domain_metrics = {}
        self.pointwise_metric_df = {}

        # Calculate model output
        self.train_inp_tar_out = self.calculate_model_outputs(forecast_data=self.train_data, forecast_dataset=self.train_dataset)
        self.val_inp_tar_out = self.calculate_model_outputs(forecast_data=self.val_data, forecast_dataset=self.val_dataset)
        self.test_inp_tar_out = self.calculate_model_outputs(forecast_data=self.test_data, forecast_dataset=self.test_dataset)
        if skip_eval:
            return

        # Evaluate within the domain
        evaluation_domains = ['train', 'val', 'test'] + list(DOMAINS[self.args.domain].items())
        for domain in evaluation_domains:
            domain_name = domain if type(domain) is str else domain[0]
            inp_tar_out = self.get_domain_data(domain=domain)
            self.domain_metrics[domain_name] = self.calculate_model_metrics(inp_tar_out, domain=domain)
            if args.create_plots:
                self.create_error_grid_plots(inp_tar_out, domain=domain)
                self.create_forecast_plots(inp_tar_out, domain=domain)

        # Evaluate across domains ...
        # ... the KPIs
        results_df = pd.DataFrame(data=[self.k], columns=['fold'])
        kpis_df = pd.concat(self.domain_metrics.values(), axis=1)
        self.results = pd.concat([results_df, kpis_df], axis=1)
        # ... plot pointwise metrics
        if args.create_plots:
            self.create_pointwise_plot(metric='loss')
            self.create_pointwise_plot(metric='RMSE')

    def calculate_model_outputs(self, forecast_data, forecast_dataset):
        self.model.eval()
        if not self.args.pytorch_forecasting:
            inputs, target = forecast_data[:]
            with torch.no_grad():
                mu, var = self.model(inputs.float().to(self.model.device))
        elif self.args.pytorch_forecasting:
            dataloader = forecast_dataset.to_dataloader(train=False, batch_size=len(forecast_dataset), shuffle=False)
            with torch.no_grad():
                for inputs, _ in dataloader:    # only 1 iteration, because of loader definition
                    mu, var = self.model(move_to(inputs, self.device))

        inputs, target = forecast_data[:]
        mu = mu.cpu()
        sigma = torch.sqrt(var).cpu()
        glucose_input = inputs[:, :, 0]
        target = target.squeeze()
        num_samples = glucose_input.size()[0]
        len_input = glucose_input.size()[1]
        len_target = target.size()[1]
        inp_tar_out_data = torch.cat((glucose_input, target, mu, sigma), 1)
        column_names = [f'input_{i}' for i in range(len_input)] + \
                       [f'target_{i}' for i in range(len_target)] + \
                       [f'output_{i}' for i in range(len_target)] + \
                       [f'sigma_{i}' for i in range(len_target)]
        data_df = pd.DataFrame(inp_tar_out_data, columns=column_names)
        info_df = pd.DataFrame([self.k] * num_samples, columns=['fold'])
        file_data_df = forecast_data.window_data[['window_id', 'file_id', 'study', 'patient_id_study', 'sensor', 'treatment', 'diabetes_type', 'diabetes_duration', 'hba1c', 'sex', 'age', 'window_start_datetime']]
        all_data_df = pd.concat([info_df,
                                 file_data_df,
                                 forecast_data.imputation_slicing_times[['first_dt_resampled_input', 'last_dt_resampled_input', 'first_dt_resampled_target', 'last_dt_resampled_target']],
                                 data_df], axis=1)
        all_data_df.rename(columns={'first_dt_resampled_input': 'first_input_dt',
                                    'last_dt_resampled_input': 'last_input_dt',
                                    'first_dt_resampled_target': 'first_target_dt',
                                    'last_dt_resampled_target': 'last_target_dt'}, inplace=True)
        return all_data_df

    def get_domain_data(self, domain):
        domain_name = domain if type(domain) is str else domain[0]

        if domain_name == 'train':
            return self.train_inp_tar_out
        elif domain_name == 'val':
            return self.val_inp_tar_out
        elif domain_name == 'test':
            return self.test_inp_tar_out
        elif type(domain) is tuple:
            relevant_entries = domain[1]
            if self.args.domain in ['diabetes_type', 'gender', 'sensor']:
                return self.test_inp_tar_out.loc[self.test_inp_tar_out[self.args.domain].isin(relevant_entries), :]
            elif self.args.domain in ['treatment']:
                train_part = self.train_inp_tar_out.loc[self.train_inp_tar_out[self.args.domain].isin(relevant_entries), :]
                val_part = self.val_inp_tar_out.loc[self.val_inp_tar_out[self.args.domain].isin(relevant_entries), :]
                test_part = self.test_inp_tar_out.loc[self.test_inp_tar_out[self.args.domain].isin(relevant_entries), :]
                return pd.concat(objs=[train_part, val_part, test_part], axis=0, ignore_index=True)
            else:
                print(f'Evaluation for the domains args.domain = {self.args.domain} is not implemented.')
                return 0
        else:
            print(f'Evaluation for the domains args.domain = {self.args.domain} is not implemented.')
            return 0

    def calculate_model_metrics(self, inp_tar_out, domain=''):
        domain_name = domain if type(domain) is str else domain[0]

        targets = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'target' in col]].iloc[:, 2:].values).float().to(self.device).unsqueeze(dim=2)
        means = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'output' in col]].values).float().to(self.device)
        sigmas = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'sigma' in col]].values).float().to(self.device)
        outputs = (means, sigmas)

        domain_name_str = domain_name + '_' if domain_name != '' else ''
        kpis = pd.DataFrame([[self.criterion(outputs, targets).item()]], columns=[domain_name_str + 'loss'])
        for m, metric_fn in self.metrics.items():
            kpis[domain_name_str + m] = None
            kpis.at[0, domain_name_str + m] = metric_fn(outputs, targets)
        return kpis

    def create_pointwise_plot(self, metric):
        metric_names = [name for name in self.results.columns if 'pointwise '+metric in name]
        num_points = len(self.results.loc[0, metric_names[0]])
        domain_names = [name.split('_')[0] for name in metric_names]
        num_domains = len(domain_names)

        points = [k * self.args.sample_step for k in range(num_points)] * num_domains
        domains = list(np.repeat(domain_names, num_points))
        values = []
        for m_name in metric_names:
            values = values + list(self.results.loc[0, m_name])
        self.pointwise_metric_df[metric] = pd.DataFrame(list(zip(points, values, domains)), columns =['point', 'value', 'domain'])

        pointwise_plots_path = os.path.join(self.args.base_dir, self.args.mtype, self.args.save_string, "plots", "pointwise")
        if not os.path.exists(pointwise_plots_path):
            os.makedirs(pointwise_plots_path)

        plot_pointwise_metric(self.pointwise_metric_df[metric], metric, save_to=pointwise_plots_path + f"/model_{self.k}_pointwise_" + metric, plot_show=self.args.plot_show)

    def create_error_grid_plots(self, inp_tar_out, domain):
        domain_name = domain if type(domain) is str else domain[0]

        targets = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'target' in col]].iloc[:, 2:].values).to(self.device)
        means = torch.tensor(inp_tar_out.loc[:, [col for col in inp_tar_out.columns if 'output' in col]].values).to(self.device)

        error_grid_plots_path = os.path.join(self.args.base_dir, self.args.mtype, self.args.save_string, "plots", "error_grids")
        if not os.path.exists(error_grid_plots_path):
            os.makedirs(error_grid_plots_path)

        plot_error_grid(means, targets, domain_name, grid_type='clarke', save_to=error_grid_plots_path + f"/model_{self.k}_CEG_" + domain_name, plot_show=self.args.plot_show)
        plot_error_grid(means, targets, domain_name, grid_type='parkes', save_to=error_grid_plots_path + f"/model_{self.k}_PEG_" + domain_name, plot_show=self.args.plot_show)

    def create_forecast_plots(self, inp_tar_out, domain, plot_window_ids=None):
        domain_name = domain if type(domain) is str else domain[0]

        if domain_name == 'train':
            forecast_dataset = self.train_data
            sample_step = forecast_dataset.sample_step
            window_data = forecast_dataset.window_data
            slicing_data = forecast_dataset.imputation_slicing_times
        elif domain_name == 'val':
            forecast_dataset = self.val_data
            sample_step = forecast_dataset.sample_step
            window_data = forecast_dataset.window_data
            slicing_data = forecast_dataset.imputation_slicing_times
        elif domain_name == 'test':
            forecast_dataset = self.test_data
            sample_step = forecast_dataset.sample_step
            window_data = forecast_dataset.window_data
            slicing_data = forecast_dataset.imputation_slicing_times
        elif type(domain) is tuple:
            relevant_entries = domain[1]
            if self.args.domain in ['diabetes_type', 'gender', 'sensor']:
                forecast_dataset = self.test_data
                sample_step = forecast_dataset.sample_step
                window_data = forecast_dataset.window_data
                slicing_data = forecast_dataset.sclicing_data
            elif self.args.domain in ['treatment']:
                sample_step = self.test_data.sample_step
                window_data_train_part = self.train_data.window_data.loc[self.train_data.window_data[self.args.domain].isin(relevant_entries), :]
                window_data_val_part = self.val_data.window_data.loc[self.val_data.window_data[self.args.domain].isin(relevant_entries), :]
                window_data_test_part = self.test_data.window_data.loc[self.test_data.window_data[self.args.domain].isin(relevant_entries), :]
                window_data = pd.concat(objs=[window_data_train_part, window_data_val_part, window_data_test_part], axis=0, ignore_index=True)
                window_ids = list(window_data.window_id)

                slicing_data_train_part = self.train_data.imputation_slicing_times.loc[self.train_data.imputation_slicing_times['window_id'].isin(window_ids), :]
                slicing_data_val_part = self.val_data.imputation_slicing_times.loc[self.val_data.imputation_slicing_times['window_id'].isin(window_ids), :]
                slicing_data_test_part = self.test_data.imputation_slicing_times.loc[self.test_data.imputation_slicing_times['window_id'].isin(window_ids), :]
                slicing_data = pd.concat(objs=[slicing_data_train_part, slicing_data_val_part, slicing_data_test_part], axis=0, ignore_index=True)
            else:
                print(f'Evaluation for the domains args.domain = {self.args.domain} is not implemented.')
        else:
            print(f'Evaluation for the domains args.domain = {self.args.domain} is not implemented.')

        if plot_window_ids is None:
            num_samples = min(len(inp_tar_out.window_id), 3)
            window_ids = random.sample(list(inp_tar_out.window_id), num_samples)
        else:
            window_ids = plot_window_ids

        forecast_plots_path = os.path.join(self.args.base_dir, self.args.mtype, self.args.save_string, "plots", "forecast")
        if not os.path.exists(forecast_plots_path):
            os.makedirs(forecast_plots_path)

        for plot_num, win_id in enumerate(window_ids):
            sample = inp_tar_out[inp_tar_out.window_id == win_id]
            window = window_data[window_data.window_id == win_id]
            slicing = slicing_data[slicing_data.window_id == win_id]
            file_id = sample.file_id.item()
            window_start_dt = sample.window_start_datetime.item()

            inputs = sample.loc[:, [col for col in inp_tar_out.columns if 'input' in col]].iloc[:, 2:].values[0]
            targets = sample.loc[:, [col for col in inp_tar_out.columns if 'target' in col]].iloc[:, 2:].values[0]
            mus = sample.loc[:, [col for col in inp_tar_out.columns if 'output' in col]].values[0]
            sigmas = sample.loc[:, [col for col in inp_tar_out.columns if 'sigma' in col]].values[0]

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

            kpis = self.calculate_model_metrics(sample)
            kpis = kpis[['loss', 'RMSE']].to_dict(orient='records')[0]

            plot_forecast(file_id, date,
                          input_datetimes, inputs,
                          target_datetimes, targets,
                          mus, sigmas,
                          measure_datetimes, measure_values,
                          domain_name,
                          kpis=kpis,
                          save_to=forecast_plots_path + f"/model_{self.k}_forecast_" + domain_name + f'_{plot_num+1}',
                          plot_show = self.args.plot_show
                          )


class AggregateEvaluator:
    def __init__(self,
                 results,
                 logs,
                 args):
        self.logs = logs
        self.args = args
        self.results = pd.DataFrame()

        self.results = pd.concat(results, ignore_index=True)
        aggr_kpis = self.results.loc[0:1].copy()
        aggr_kpis.fold = ['mean', 'std']
        for column in aggr_kpis.columns[1:]:
            array = np.stack(self.results[column].values)
            aggr_kpis.at[0, column] = array.mean(axis=0)
            aggr_kpis.at[1, column] = array.std(axis=0)
        self.results = pd.concat([self.results, aggr_kpis], ignore_index=True)
        args_df = pd.DataFrame.from_dict(self.args, orient='index').transpose()
        args_df = pd.concat([args_df]*len(self.results), ignore_index=True)
        self.results = pd.concat([self.results, args_df], axis=1)

        self.results.to_excel(os.path.join(args.base_dir, args.mtype, args.save_string, "results.xlsx"), index=False, float_format="%.3f")

        if args.create_plots:
            visualize_model_performance(logs, args)
            # self.create_pointwise_plot(metric='loss')
            # self.create_pointwise_plot(metric='RMSE')

    def create_pointwise_plot(self, metric):
        dfs = []
        for k in range(len(self.model_evaluations)):
            dfs.append(self.model_evaluations[0].pointwise_metric_df[metric])
        df = pd.concat(dfs, ignore_index=True)
        df = df.loc[(df.domain == 'train') | (df.domain == 'val')]

        pointwise_plots_path = os.path.join(self.args.base_dir, self.args.mtype, self.args.save_string, "plots", "pointwise")
        plot_pointwise_metric(df, metric, save_to=pointwise_plots_path + f"/pointwise_" + metric, plot_show=self.args.plot_show)
