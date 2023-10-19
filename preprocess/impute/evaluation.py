import pandas as pd
import numpy as np
import torch

from preprocess.impute.inference import gp_sampling


def calculate_batch_kpis(input_batch, target_batch, lengths, model, sample_length, sample_step=5):
    device = model.device
    batch_size = len(input_batch)
    batch_kpis = {'RMSE': [], 'mean_volatility': [], 'variance_volatility': [], 'variance_avg': [], 'num_variance_collapses': []}
    for window_num in range(batch_size):
        input, target = input_batch[window_num][:lengths[window_num]], target_batch[window_num][:lengths[window_num]]
        mean_at_points, variance_at_points = gp_sampling(input, target, model, sample_length=sample_length, sample_step=sample_step, inputs_gp=input)
        mean, variance = gp_sampling(input, target, model, sample_length=sample_length, sample_step=sample_step)
        mean_at_points, variance_at_points = mean_at_points.to(device), variance_at_points.to(device)
        mean, variance = mean.to(device), variance.to(device)

        rmse = torch.sqrt(torch.mean((target-mean_at_points)**2)).item()
        mean_volatility = torch.std(mean).item()
        variance_volatility = torch.std(variance).item()
        variance_avg_size = torch.mean(variance).item()
        num_variance_collapse = variance[variance < 0.1].numel()

        batch_kpis['RMSE'].append(rmse)
        batch_kpis['mean_volatility'].append(mean_volatility)
        batch_kpis['variance_volatility'].append(variance_volatility)
        batch_kpis['variance_avg'].append(variance_avg_size)
        batch_kpis['num_variance_collapses'].append(num_variance_collapse)

    batch_kpis_df = pd.DataFrame.from_dict(batch_kpis)
    aggr_batch_kpis_dict = aggregate_kpis(batch_kpis_df)
    return aggr_batch_kpis_dict


def aggregate_kpis(kpis_df, train_val=None):
    rmse = np.sqrt((kpis_df.RMSE ** 2).sum() / len(kpis_df))
    mean_vola = np.sqrt((kpis_df.mean_volatility ** 2).sum() / len(kpis_df))
    variance_vola = np.sqrt((kpis_df.variance_volatility ** 2).sum() / len(kpis_df))
    variance_avg = kpis_df.variance_avg.mean()
    num_variance_collapses = kpis_df['num_variance_collapses'].sum()
    aggr_kpis_dict = {'RMSE': rmse,
                      'mean_volatility': mean_vola,
                      'variance_volatility': variance_vola,
                      'variance_avg': variance_avg,
                      'num_variance_collapses': num_variance_collapses,
                      }
    if train_val is not None:
        aggr_kpis_dict = {train_val + '_' + key: value for key, value in aggr_kpis_dict.items()}
    return aggr_kpis_dict
