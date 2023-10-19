import itertools
import os
import sys
import warnings
import json
import time
import random
import torch

import pandas as pd
import numpy as np

from forecast.data import get_data_location, load_data
from forecast.helper import get_metrics, get_live_metrics, get_model
from forecast.train import train_model
from forecast.evaluation import ModelEvaluator, AggregateEvaluator
from forecast.config import args
from preprocess.impute.config import args as gp_args
from utils.setup import init_gpu

sys.path.append(os.path.expanduser("~/Thesis/icarus/"))
warnings.filterwarnings("ignore", message="Negative variance values detected. This is likely due to numerical instabilities. Rounding negative variances up to 1e-06.")
pd.set_option('display.float_format', lambda x: '%.9f' % x)
np.set_printoptions(suppress=True)


similarity_measure = sys.argv[1]
lambda_ols = float(sys.argv[2])
lambda_l1 = float(sys.argv[3])
seed = int(sys.argv[4])
gpu = int(sys.argv[5])

print('similarity_measure', similarity_measure)
print('lambda_ols', lambda_ols)
print('lambda_l1', lambda_l1)
print('seed', seed)
print('gpu', gpu)

config = {
    'model': 'Layer',
    'loss': 'LayerLoss',
    'feature_extractor': 'LSTM',
    'fine_tuning': True,
    'fe_dir': 'save/Thesis/Layer_Experiment/LSTM_single',
    'similarity_measure_name': similarity_measure,                       # MMD, CS
    'num_gdus': {0: 9, 1: 9, 2: 9, 3: 9},
    'domain_dim': 10,
    'sigmas_path': os.path.join(args.base_dir, args.mtype, 'save/Thesis/Layer_Experiment/avg_sigmas.xlsx'),

    'epochs': 10000,
    'early_stopping': 200,
    'target_duration': 1,
    # 'target_durations': [0.5, 1, 2],
    'split_attribute': 'treatment',

    'seed': seed,
    'gpu': f'{gpu}',
    'plot_show': False,
    'save_models': False,
    'create_plots': False,
}


args.model = config['model']
args.loss = config['loss']
args['Layer_hps']['feature_extractor'] = config['feature_extractor']
args['Layer_hps']['similarity_measure_name'] = config['similarity_measure_name']
args['Layer_hps']['domain_dim'] = config['domain_dim']
args['Layer_hps']['fine_tuning_FEs_path'] = None if not config['fine_tuning'] else True
args.epochs = config['epochs']
args.target_duration = config['target_duration']
# args.target_duration = config['target_durations']
gp_args.split_attribute = config['split_attribute']
gp_args.K = 4
args.seed = config['seed']
args.gpu = config['gpu']
args.create_plots = config['create_plots']
args.plot_show = config['plot_show']
args.save_models = config['save_models']
args.pytorch_forecasting = True if config['feature_extractor'] == 'TFT' else False

sigmas = pd.read_excel(config['sigmas_path'])

sys.path.append(os.path.expanduser(args.repo_location))

torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed)
random.seed(args.seed)
device = init_gpu(args.gpu)

ft_e2e = 'FT' if config['fine_tuning'] else 'E2E'
folder_name = f"Layers_{args['Layer_hps']['feature_extractor']}_{ft_e2e}_{config['similarity_measure_name']}_{args.seed}_"
args.save_dir = os.path.join(args.base_dir, args.mtype, folder_name)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.txt'), 'w') as file:
        json.dump(config, file, indent=4)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as file:
        json.dump(args, file, indent=4)
    with open(os.path.join(args.save_dir, 'gp_args.json'), 'w') as file:
        json.dump(gp_args, file, indent=4)

print(f"\n Lambda_OLS = {lambda_ols}, Lambda_L1 =  {lambda_l1}\n")
horizon = config['target_duration']
args['Layer_hps']['lambda_OLS'] = lambda_ols
args['Layer_hps']['lambda_sparse'] = lambda_l1
args.save_string = folder_name + f"/{args.model}_{args.seed}_{horizon}_{gp_args.split_attribute}_lambdaOLS{lambda_ols}_lambda_l1{lambda_l1}_{time.strftime('%Y%m%d-%H%M%S')}"
data_path, folds = get_data_location(args, gp_args, device)
args.data_path = data_path
args.folds = folds

if config['fine_tuning']:
    fe_path = os.path.join(args.base_dir, args.mtype, config['fe_dir'])
    seed_folder = [f_name for f_name in os.listdir(fe_path) if f'_{args.seed}_' in f_name][0]
    seed_path = os.path.join(fe_path, seed_folder)
    horizon_folder = [f_name for f_name in os.listdir(seed_path) if f'_{args.seed}_{horizon}_' in f_name][0]
    args['Layer_hps']['fine_tuning_FEs_path'] = os.path.join(config['fe_dir'], seed_folder, horizon_folder)

if not os.path.exists(os.path.join(args.base_dir, args.mtype, args.save_string)):
    os.makedirs(os.path.join(args.base_dir, args.mtype, args.save_string))
with open(os.path.join(args.base_dir, args.mtype, args.save_string, 'config.json'), 'w') as file:
    json.dump({**args, **gp_args, **args}, file, indent=4)
with open(os.path.join(args.base_dir, args.mtype, args.save_string, 'args.json'), 'w') as file:
    json.dump(args, file, indent=4)
with open(os.path.join(args.base_dir, args.mtype, args.save_string, 'gp_args.json'), 'w') as file:
    json.dump(gp_args, file, indent=4)

logs, results = {}, []
for k in folds:
    print("\nData Split: ", k)

    # Different number of GDUs for different folds!
    args.Layer_hps['num_gdus'] = config['num_gdus'][k]
    args.Layer_hps['sigma'] = sigmas.loc[(sigmas.horizon == horizon) & (sigmas.fold == k), 'sigma'].iloc[0]

    # load data, model & metrics
    train_data, val_data, test_data, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(k, data_path, args)
    model, optimiser, scheduler = get_model(args, device, train_data, train_dataset, num_input_features=train_data[0][0].size(1), input_length=train_data[0][0].size(0), output_size=train_data[0][-1].size(0), fold=k)
    criterion, metrics = get_live_metrics(args, train_data, device)
    # train
    model, logs[k] = train_model(model, criterion, optimiser, scheduler, metrics, train_loader, val_loader, test_loader, args, k)
    del train_loader, val_loader, test_loader, optimiser, scheduler, criterion, metrics
    torch.cuda.empty_cache()
    # evaluate and save
    criterion, metrics = get_metrics(args, train_data, device)
    model_evaluation = ModelEvaluator(model,
                                      train_data, val_data, test_data,
                                      train_dataset, val_dataset, test_dataset,
                                      criterion, metrics, args, k, logs)
    results.append(model_evaluation.results)
    del train_data, val_data, test_data, train_dataset, val_dataset, test_dataset, model, model_evaluation, criterion, metrics
    torch.cuda.empty_cache()
AggregateEvaluator(results, logs, args)

sys.exit()
