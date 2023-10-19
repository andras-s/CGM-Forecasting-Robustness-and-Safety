import os
import sys
import random
import json
import torch

from forecast.data import get_data_location
from forecast.loss import NLL, NLLMetric, PointwiseNLL, LastNLL, PEGMetric, LastPEG, NLLPEGSurface, NLLPEGMetric, LastNLLPEG, RMSEMetric, LastRMSE, PointwiseRMSE, MAE, PointwiseMAE, LayerLoss, CEGPercentages, PEGLoss, PEGPercentages, PointwiseCEGPercentages, PointwisePEGPercentages, LastPEGPercentages
from forecast.model import t_0, RegressionLSTM, RegressionConvTransformer, RegressionTFT, EnsembleModel, LayerModel, RegressionBaseline, ClassificationLSTM
from utils.setup import init_gpu


def get_metrics(args, train_data, device):
    if args.mtype == 'regression':
        criterion = NLL()
        metrics = {
            'RMSE':             RMSEMetric(),
            'MAE':              MAE(),
            'NLL':              NLLMetric(),
            'PEG':              PEGMetric(),
            # 'TimeLag':          TimeLag(),
            # 'MAPE':             MAPE(),
            # 'R2':               R2(),
            # 'MSE':              MSE(),
            # 'gMSE':             gMSE(),
            # 'MAD':              MAD(),
            # 'MARD':             MARD(),
            'CEG [% in A-E]':   CEGPercentages(),
            'PEG [% in A-E]':   PEGPercentages(),
            'pointwise loss':   PointwiseNLL(),
            'pointwise RMSE':   PointwiseRMSE(),
            'pointwise MAE':   PointwiseMAE(),
            'pointwise NLL':   PointwiseNLL(),
            'pointwise CEG [% in A-E]': PointwiseCEGPercentages(),
            'pointwise PEG [% in A-E]': PointwisePEGPercentages(),
        }
    metrics = {k: v.to(device) for k, v in metrics.items()}
    return criterion, metrics


def get_live_metrics(args, train_data, device):
    if args.mtype == 'regression':
        if args.loss == 'NLL':
            criterion = NLL()
        elif args.loss == 'NLLPEGSurface':
            criterion = NLLPEGSurface(args.PEG_weight)
        elif args.loss == 'LayerLoss':
            criterion = LayerLoss(device=device,
                                  criterion=NLL(),
                                  sigma=args.Layer_hps['sigma'],
                                  lambda_OLS=args.Layer_hps['lambda_OLS'],
                                  lambda_orth=args.Layer_hps['lambda_orth'],
                                  lambda_sparse=args.Layer_hps['lambda_sparse'],
                                  orthogonal_loss=args.Layer_hps['orthogonal_loss'],
                                  sparse_coding=args.Layer_hps['sparse_coding'])
        if args.extensive_logging:
            metrics = {
                'NLL': NLLMetric(),
                'last NLL': LastNLL(),
                'NLLPEGSurface': NLLPEGMetric(args.PEG_weight),
                'last NLLPEGSurface': LastNLLPEG(args.PEG_weight),
                'PEG': PEGMetric(),
                'last PEG': LastPEG(),
                'RMSE': RMSEMetric(),
                'last RMSE': LastRMSE(),
                'PEG [% in A-E]': PEGPercentages(),
                'last PEG [% in A-E]': LastPEGPercentages(),
                }
        else:
            metrics = {'RMSE'               : RMSEMetric(),
                       'pointwise loss'     : PointwiseNLL(),
                       'pointwise RMSE'     : PointwiseRMSE(),
                       }
    criterion = criterion.to(device)
    metrics = {k: v.to(device) for k, v in metrics.items()}
    return criterion, metrics


def get_model(args, device, train_data, train_dataset, num_input_features, input_length, output_size, fold):
    if not args.pytorch_forecasting:
        if args.mtype == 'regression':
            if args.model == 't_0':
                model = t_0(device=device, train_data=train_data)
                return model.to(device), None, None
            elif args.model == 'LSTM':
                model = RegressionLSTM(
                    device=device,
                    num_input_features=num_input_features,
                    hidden_size=args.LSTM_hps['hidden'],
                    num_layers=args.LSTM_hps['num_layers'],
                    lin_layers= args.LSTM_hps['lin_layers'],
                    dropouts= args.LSTM_hps['dropouts'],
                    output_size=output_size
                )
            elif args.model == 'ConvT':
                model = RegressionConvTransformer(
                    device=device,
                    n_time_series=num_input_features,
                    forecast_history=input_length,
                    output_size=output_size,
                    sub_len=args.ConvT_hps['sub_len'],
                    q_len=args.ConvT_hps['q_len'],
                    n_embd=args.ConvT_hps['n_embd'],
                    n_head=args.ConvT_hps['n_head'],
                    num_layer=args.ConvT_hps['num_layer'],
                    scale_att=args.ConvT_hps['scale_att'],
                    dropout=args.ConvT_hps['dropout'],
                    lin_layers=args.ConvT_hps['lin_layers'],
                    lin_dropouts=args.ConvT_hps['lin_dropouts'],
                    additional_params=args.ConvT_hps['additional_params']
                )
            elif args.model == 'Ensemble':
                if args.Ensemble_hps['feature_extractor'] == 'LSTM':
                    lstm_model = RegressionLSTM(
                        device=device,
                        num_input_features=num_input_features,
                        hidden_size=args.LSTM_hps['hidden'],
                        num_layers=args.LSTM_hps['num_layers'],
                        lin_layers= args.LSTM_hps['lin_layers'],
                        dropouts= args.LSTM_hps['dropouts'],
                        output_size=output_size
                    )
                    if args.Ensemble_hps['fine_tuning_FEs_path'] is not None:
                        lstm_model.load_state_dict(torch.load(os.path.join(args.base_dir, args.mtype, args.Ensemble_hps['fine_tuning_FEs_path'], "models", f"weights_{fold}.pth")))
                        for param in lstm_model.parameters():
                            param.requires_grad = False

                    feature_extractor = lstm_model.lstm
                    model = EnsembleModel(
                        feature_extractor=feature_extractor,
                        feature_vector_size=args.LSTM_hps['hidden'],
                        output_size=output_size,
                        num_heads=args.Ensemble_hps['num_heads']
                    )

            elif args.model == 'Layer':
                if args.Layer_hps['feature_extractor'] == 'LSTM':
                    lstm_model = RegressionLSTM(
                        device=device,
                        num_input_features=num_input_features,
                        hidden_size=args.LSTM_hps['hidden'],
                        num_layers=args.LSTM_hps['num_layers'],
                        lin_layers=args.LSTM_hps['lin_layers'],
                        dropouts=args.LSTM_hps['dropouts'],
                        output_size=output_size
                    )
                    if args.Layer_hps['fine_tuning_FEs_path'] is not None:
                        lstm_model.load_state_dict(torch.load(os.path.join(args.base_dir, args.mtype, args.Layer_hps['fine_tuning_FEs_path'], "models", f"weights_{fold}.pth")))
                        for param in lstm_model.parameters():
                            param.requires_grad = False

                    feature_extractor = lstm_model.lstm
                    model = LayerModel(
                        device=device,
                        task='probabilistic_forecasting',
                        feature_extractor=feature_extractor,
                        feature_vector_size=args.LSTM_hps['hidden'],
                        output_size=output_size,
                        num_gdus=args.Layer_hps['num_gdus'],
                        domain_dim=args.Layer_hps['domain_dim'],
                        sigma=args.Layer_hps['sigma'],
                        similarity_measure_name=args.Layer_hps['similarity_measure_name'],
                        softness_param=args.Layer_hps['softness_param']
                    )

        elif args.mtype == 'classification':
            model = ClassificationLSTM(device=device,
                                       num_input_features=num_input_features,
                                       hidden_size=args.LSTM_hps['hidden'],
                                       num_layers=args.LSTM_hps['num_layers'],
                                       output_size=output_size)

    elif args.pytorch_forecasting:
        if args.mtype == 'regression':
            if args.model == 'TFT':
                model = RegressionTFT(hidden_size = args.TFT_hps['hidden_size'],
                                      attention_head_size=args.TFT_hps['attention_head_size'],
                                      dropout=args.TFT_hps['dropout'],
                                      hidden_continuous_size=args.TFT_hps['hidden_continuous_size'],
                                      device=device,
                                      train_timeseries_dataset=train_dataset)
            elif args.model == 'Baseline':
                model = RegressionBaseline().to(device)
                model.device = device
                return model, None, None

    model = model.to(device)
    # model = DataParallel(model, device_ids=args.gpu_ids)
    model.device = device
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=args.gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=args.factor, patience=args.patience)

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, optimiser, scheduler


def initialize_run():
    from forecast.config import args
    from preprocess.impute.config import args as gp_args
    init_args = args
    init_gp_args = gp_args

    sys.path.append(os.path.expanduser(init_args.repo_location))  # append repo location to $PYTHONPATH such that imports work

    torch.backends.cudnn.enabled = False
    torch.manual_seed(init_args.seed)
    random.seed(init_args.seed)
    device = init_gpu(init_args.gpu)

    data_path, folds = get_data_location(args, gp_args, device)
    init_args['data_path'] = data_path
    init_gp_args['folds'] = folds

    if not os.path.exists(os.path.join(init_args.base_dir, init_args.mtype, init_args.save_string)):
        os.makedirs(os.path.join(init_args.base_dir, init_args.mtype, init_args.save_string))
    with open(os.path.join(init_args.base_dir, init_args.mtype, init_args.save_string, 'config.json'), 'w') as file:
        json.dump({**init_args, **init_gp_args, **init_args}, file, indent=4)
    with open(os.path.join(init_args.base_dir, init_args.mtype, init_args.save_string, 'args.json'), 'w') as file:
        json.dump(init_args, file, indent=4)
    with open(os.path.join(init_args.base_dir, init_args.mtype, init_args.save_string, 'gp_args.json'), 'w') as file:
        json.dump(init_gp_args, file, indent=4)

    return data_path, folds, device, init_args, init_gp_args


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.float().to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res


def get_all_inputs_targets(data_loader, args, device):
    # Should be called with a data loader with batch_size = len(val_data)
    for inputs, targets in data_loader:
        if not args.pytorch_forecasting:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
        elif args.pytorch_forecasting:
            inputs, targets = move_to(inputs, device), targets[0].unsqueeze(2).to(device)
    return inputs, targets


def calc_loss(output, target, criterion, args, model):
    if args.loss == 'LayerLoss':
        loss = criterion(output, target, model)
    else:
        loss = criterion(output, target)
    return loss


def save_intermediate_best_model(model, args, fold, epoch):
    intermediate_models_path = os.path.join(args.base_dir, args.mtype, args.save_string, "models", "intermediate")
    if not os.path.exists(intermediate_models_path):
        os.makedirs(intermediate_models_path)
    torch.save(model.state_dict(), intermediate_models_path + f"/fold_{fold}_epoch_{epoch}.pth")
