from forecast.data import load_data
from forecast.helper import initialize_run, get_metrics, get_live_metrics, get_model
from forecast.train import train_model
from forecast.evaluation import ModelEvaluator, AggregateEvaluator

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Negative variance values detected. This is likely due to numerical instabilities. Rounding negative variances up to 1e-06.")
pd.set_option('display.float_format', lambda x: '%.9f' % x)
np.set_printoptions(suppress=True)


def main():
    """This function handles the training and evaluation of forecasting models"""
    data_path, folds, device, args, gp_args = initialize_run()
    logs, results = {}, []

    for k in folds:
        print("\nData Split: ", k)
        # load data, model & metrics
        train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(k, data_path, args)
        model, optimiser, scheduler = get_model(args, device, train_data, train_dataset, num_input_features=train_data[0][0].size(1), input_length=train_data[0][0].size(0), output_size=train_data[0][-1].size(0), fold=k)
        criterion, metrics = get_live_metrics(args, train_data, device)
        # train
        model, logs[k] = train_model(model, criterion, optimiser, scheduler, metrics, train_loader, val_loader, test_loader, args, k)
        # evaluate and save
        criterion, metrics = get_metrics(args, train_data, device)
        model_evaluation = ModelEvaluator(model,
                                          train_data, val_data, test_data,
                                          train_dataset, val_dataset, test_dataset,
                                          criterion, metrics, args, k, logs)
        results.append(model_evaluation.results)
    AggregateEvaluator(results, logs, args)


if __name__ == '__main__':
    main()
