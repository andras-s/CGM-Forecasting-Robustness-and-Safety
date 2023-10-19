import os
import random
import numpy as np
import pandas as pd
import json
import warnings

import torch
import gpytorch
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.expanduser("~/Thesis/icarus/"))    # append repo location to $PYTHONPATH such that imports work

from preprocess.impute.data import GaussianProcessDataset
from preprocess.impute.model import GaussianProcess
from preprocess.impute.train import train_model
from preprocess.impute.inference import gp_sampling
from preprocess.impute.config import args

from utils.setup import init_gpu
from utils.plot import create_gp_training_plots
from utils.data.load import read_data
from utils.data.manipulate import create_windows, split_by_patient, generate_predefined_split


# suppress NumericalWarnings occurring during the first few training iterations
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", message="CG terminated in 1000 iterations ")
warnings.filterwarnings("ignore", message="The input matches the stored training data")
warnings.filterwarnings("ignore", message="A not p.d., added jitter of 1.0e-04 to the diagonal")
warnings.filterwarnings("ignore", message="A not p.d., added jitter of 1.0e-05 to the diagonal")
warnings.filterwarnings("ignore", message="A not p.d., added jitter of 1.0e-06 to the diagonal")
warnings.filterwarnings("ignore", message="Negative variance values detected. This is likely due to numerical instabilities. Rounding negative variances up to 1e-06.")
warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")


def main():
    assert args.kernel in ("RBF", "Matern-12", "Matern-32", "Matern-52", "SM"), f"You passed an invalid kernel, namely {args.kernel}"
    # assert len(args.gpu) = 1,  "You have selected more than one GPU card. Currently only GP regression with one GPU card is supported."

    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = init_gpu(args.gpu)

    # read data
    data, patients = read_data(load_dir=args.load_dir, sensors=args.sensors)                                              # todo: patients = list of IDs (with AP data compatible?)
    windows = create_windows(data, patients, load_dir=args.load_dir, args=args, window_length=args.window_length, window_start=args.window_start)
    data, patients, foldid = split_by_patient(data=windows, K=args.K, seed=args.seed)
    data_generator = generate_predefined_split(data, patients, foldid)

    # logging
    logs = pd.DataFrame()

    if not os.path.exists(os.path.join(args.save_dir, args.save_string, "plots/exemplary_inference")):
        os.makedirs(os.path.join(args.save_dir, args.save_string, "plots/exemplary_inference"))
    if not os.path.exists(os.path.join(args.save_dir, args.save_string, "models")):
        os.makedirs(os.path.join(args.save_dir, args.save_string, "models"))

    with open(os.path.join(args.save_dir, args.save_string, 'config.json'), 'w') as file:
        json.dump(args, file, indent=4)

    for k, ((train_df, val_df), _) in enumerate(data_generator):
        print("Cross-validation fold: ", k)

        # load data
        train_data = GaussianProcessDataset(train_df)
        val_data = GaussianProcessDataset(val_df)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

        train_x_init, train_y_init = next(iter(train_loader))[0][0], next(iter(train_loader))[1][0]

        # model and optimizer
        if args.noise_fixed_train != 0:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            likelihood.noise = torch.tensor(args.noise_fixed_train)
            likelihood.raw_noise.requires_grad_(False)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(args.noise_lower_limit_train, args.noise_upper_limit_train)).to(device)
        model = GaussianProcess(train_x_init, train_y_init, likelihood, args).to(device)     # initialize with no data info
        model.device = device
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimiser = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=args.gamma)

        # metrics
        criterion = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        metrics = {'log-loss': np.log10}

        # train
        best_model, logs_fold = train_model(likelihood, model, args.kernel, criterion, optimiser, scheduler, metrics,
                                            train_loader, val_loader, args, k)
        logs = pd.concat([logs, logs_fold])

        # plot example GP sampling for each 0th item in the batch
        for b, (inputs, target, lengths) in enumerate(train_loader):
            if b % 1 == 0:
                inputs, target = inputs[0][:lengths[0]], target[0][:lengths[0]]
                gp_sampling(inputs, target, best_model, sample_length=args.window_length, sample_step=5,
                            plot_style=args.plot_style,
                            save_plot=os.path.join(args.save_dir, args.save_string, f"plots/exemplary_inference/train_{k+1}_{b+1}"))

        for b, (inputs, target, lengths) in enumerate(val_loader):
            if b % 1 == 0:
                inputs, target = inputs[0][:lengths[0]], target[0][:lengths[0]]
                gp_sampling(inputs, target, best_model, sample_length=args.window_length, sample_step=5,
                            plot_style=args.plot_style,
                            save_plot=os.path.join(args.save_dir, args.save_string, f"plots/exemplary_inference/val_{k+1}_{b+1}"))

        torch.save(best_model.state_dict(), args.save_dir + args.save_string + f"/models/weights_{k+1}.pth")

    results = logs[logs.batch.isna()]
    best_result_indices = results.groupby(["fold"])["val_loss"].transform(min) == results['val_loss']
    results = results[best_result_indices]
    logs.to_excel(os.path.join(args.save_dir, args.save_string, "logs.xlsx"), index=False)
    results.to_excel(os.path.join(args.save_dir, args.save_string, "results.xlsx"), index=False)

    create_gp_training_plots(logs, os.path.join(args.save_dir, args.save_string, "plots"))


if __name__ == '__main__':
    main()
