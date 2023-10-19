import os
import sys
import warnings
import json
import random
import torch

import pandas as pd
import numpy as np

from forecast.data import get_data_location, load_data
from forecast.helper import get_live_metrics, get_model
from forecast.train import train_model_measure_runtime
from forecast.config import args
from preprocess.impute.config import args as gp_args
from utils.setup import init_gpu

sys.path.append(os.path.expanduser("~/Thesis/icarus/"))
warnings.filterwarnings("ignore", message="Negative variance values detected. This is likely due to numerical instabilities. Rounding negative variances up to 1e-06.")
pd.set_option('display.float_format', lambda x: '%.9f' % x)
np.set_printoptions(suppress=True)

model = sys.argv[1]
loss = sys.argv[2]
fold = sys.argv[3]
horizon = float(sys.argv[4]) if float(sys.argv[4]) == 0.5 else int(sys.argv[4])
seed = int(sys.argv[5])
gpu = int(sys.argv[6])

# model = "ConvT"
# loss = "NLLPEGSurface"
# fold = 0
# horizon = 1
# seed = 0
# gpu = 0

# print('model', model)
# print('loss', loss)
# print('horizon', horizon)
# print('seed', seed)
# print('gpu', gpu)


config = {
    'model': model,
    'loss': loss,

    'epochs': 10000, # todo
    'early_stopping': 200,
    'target_duration': horizon,
    'split_attribute': None, # todo

    'seed': seed,
    'gpu': f'{gpu}',
    'plot_show': False,
    'save_models': False,
    'create_plots': False,
}


args.model = config['model']
args.loss = config['loss']

args.epochs = config['epochs']
args.target_duration = config['target_duration']
gp_args.split_attribute = config['split_attribute']
gp_args.K = 5
args.seed = config['seed']
args.gpu = config['gpu']
args.create_plots = config['create_plots']
args.plot_show = config['plot_show']
args.save_models = config['save_models']
args.pytorch_forecasting = False

sys.path.append(os.path.expanduser(args.repo_location))

torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed)
random.seed(args.seed)
device = init_gpu(args.gpu)

# data_path, folds = get_data_location(args, gp_args, device)
# print("data_path: ", data_path, type(data_path))
# print("folds: ", folds, type(folds))
data_path, folds = "data_on_server/v_2/4_imputed_data/20220812-133054", [0, 1, 2, 3, 4]
args.data_path = data_path
args.folds = folds

folder_name = f"Discussion/runtimes/{args.model}_{args.loss}"
args.save_dir = folder_name
args.save_string = folder_name

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.txt'), 'w') as file:
        json.dump(config, file, indent=4)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as file:
        json.dump(args, file, indent=4)
    with open(os.path.join(args.save_dir, 'gp_args.json'), 'w') as file:
        json.dump(gp_args, file, indent=4)

k = fold
print("\nData Split: ", k)

# load data, model & metrics
train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(k, data_path, args)
model, optimiser, scheduler = get_model(args, device, train_data, train_dataset, num_input_features=train_data[0][0].size(1), input_length=train_data[0][0].size(0), output_size=train_data[0][-1].size(0), fold=k)
criterion, metrics = get_live_metrics(args, train_data, device)
# train
logs = train_model_measure_runtime(model, criterion, optimiser, scheduler, metrics, train_loader, val_loader, test_loader, args, k)

del train_loader, val_loader, test_loader, optimiser, scheduler, criterion, metrics, train_data, val_data, test_data, train_dataset, val_dataset, test_dataset
torch.cuda.empty_cache()

# runtimes = {k: logs[k]["train"]["runtime"] for k in logs.keys()}
# nums_epochs = {k: len(runtimes[k]) for k in logs.keys()}
# all_runtimes = []
# [all_runtimes.extend(runtimes[k]) for k in logs.keys()]
# avg_epoch_runtimes = {k: np.mean(runtimes[k]) for k in logs.keys()}
# mean_epoch_runtimes = {k: np.std(runtimes[k]) for k in logs.keys()}
# total_runtimes = {k: sum(runtimes[k]) for k in logs.keys()}
#
# runtimes = {'avg_num_epochs': np.mean(list(nums_epochs.values())),
#             'std_num_epochs': np.std(list(nums_epochs.values())),
#             'avg_duration_epoch': np.mean(all_runtimes),
#             'std_duration_epoch': np.std(all_runtimes),
#             'avg_total_runtime': np.mean(list(total_runtimes.values())),
#             'std_total_runtime': np.std(list(total_runtimes.values())),
#             }
runtimes = logs["train"]["runtime"]
print(runtimes)
import json
with open(os.path.join(args.save_dir, f"runtimes_fold_{k}.txt"), 'w') as file:
    file.write(json.dumps(runtimes))

sys.exit()
