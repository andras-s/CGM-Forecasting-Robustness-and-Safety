import os
import sys
import pickle
import copy
import time

from tqdm import tqdm
import numpy as np
import torch

from forecast.helper import move_to, get_all_inputs_targets, calc_loss, save_intermediate_best_model


def train_model(model, criterion, optimiser, scheduler, metrics, train_loader, val_loader, test_loader, args, k):
    if args['lazy_learner']:
        return model, {}

    logs = {i: {j: [] for j in ('loss',) + tuple(metrics.keys())} for i in ('train', 'val', 'test')} # epoch logs

    min_loss = np.inf
    epochs_since_best = 0
    best_model = None
    device = model.device

    for e in tqdm(range(args.epochs)):

        batch_logs = {i: {j: [] for j in logs[i].keys()} for i in logs.keys()}

        # training
        model.train()
        for inputs, target in train_loader:
            if not args.pytorch_forecasting:
                inputs, target = inputs.float().to(device), target.float().to(device)
            elif args.pytorch_forecasting:
                inputs, target = move_to(inputs, device), target[0].unsqueeze(2).to(device)

            optimiser.zero_grad()
            output = model(inputs)
            if args.loss == 'LayerLoss':
                loss = criterion(output, target, model)
            else:
                loss = criterion(output, target)

            batch_logs['train']['loss'].append(loss.item())

            for m, metric_fn in metrics.items():
                batch_logs['train'][m].append(metric_fn(output, target))

            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimiser.step()

        optimiser.zero_grad()
        for metric in logs['train'].keys():
            logs['train'][metric].append(np.mean(batch_logs['train'][metric], 0))

        # Evaluation
        model.eval()
        with torch.no_grad():
            for split_name, data_loader in [('val', val_loader), ('test', test_loader)]:
                inputs, target = get_all_inputs_targets(data_loader, args, device)
                output = model(inputs)
                loss = calc_loss(output, target, criterion, args, model)
                logs[split_name]['loss'].append(loss.item())
                for m, metric_fn in metrics.items():
                    logs[split_name][m].append(metric_fn(output, target))

        # check if last step led to improved model and check early-stopping criterion
        if logs['val']['loss'][-1] < min_loss:
            epochs_since_best = 0
            min_loss = logs['val']['loss'][-1]
            best_model = copy.deepcopy(model)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.early_stopping:
                print("Early stopping!")
                break

        # Save intermediate model if extensive logging is activated
        if args.extensive_logging:
            if e % 20 == 0:
                save_intermediate_best_model(best_model, args, k, e)

        for param_group in optimiser.param_groups:
            print(f"Loss: Train: {round(logs['train']['loss'][-1], 2)} Val: {round(logs['val']['loss'][-1], 2)} Test: {round(logs['test']['loss'][-1], 2)} Best Val: {round(min_loss, 2)},   Epochs since: {epochs_since_best}/{args.early_stopping}   LR: {round(param_group['lr'], 6)}")

        # Adjust LR
        if args.lr_scheduler == 'ExponentialLR':
            scheduler.step()
        elif args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(logs['val']['loss'][-1])

    if epochs_since_best < args.early_stopping:
        print("Reached maximum number of epochs!")

    # save model
    models_path = os.path.join(args.base_dir, args.mtype, args.save_string, "models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if args.save_models:
        torch.save(best_model.state_dict(), models_path + f"/weights_{k}.pth")
    # save logs
    logs_path = os.path.join(args.base_dir, args.mtype, args.save_string, "logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(logs_path + f"/logs_{k}.pkl", "wb") as f:
        pickle.dump(logs, f)

    return best_model, logs


def train_model_measure_runtime(model, criterion, optimiser, scheduler, metrics, train_loader, val_loader, test_loader, args, k):
    logs = {i: {j: [] for j in ('loss', 'runtime')} for i in ('train', 'val', 'test')} # epoch logs

    min_loss = np.inf
    epochs_since_best = 0
    device = model.device

    for e in tqdm(range(args.epochs)):

        start = time.time()
        batch_logs = {i: {j: [] for j in logs[i].keys()} for i in logs.keys()}

        # training
        model.train()
        for inputs, target in train_loader:
            if not args.pytorch_forecasting:
                inputs, target = inputs.float().to(device), target.float().to(device)
            elif args.pytorch_forecasting:
                inputs, target = move_to(inputs, device), target[0].unsqueeze(2).to(device)

            optimiser.zero_grad()
            output = model(inputs)
            if args.loss == 'LayerLoss':
                loss = criterion(output, target, model)
            else:
                loss = criterion(output, target)

            batch_logs['train']['loss'].append(loss.item())

            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimiser.step()

        optimiser.zero_grad()

        logs['train']['loss'].append(np.mean(batch_logs['train']['loss'], 0))
        end = time.time()
        logs['train']['runtime'].append(end - start)

        # Evaluation
        model.eval()
        with torch.no_grad():
            for split_name, data_loader in [('val', val_loader)]:
                inputs, target = get_all_inputs_targets(data_loader, args, device)
                output = model(inputs)
                loss = calc_loss(output, target, criterion, args, model)
                logs[split_name]['loss'].append(loss.item())

        # check if last step led to improved model and check early-stopping criterion
        if logs['val']['loss'][-1] < min_loss:
            epochs_since_best = 0
            min_loss = logs['val']['loss'][-1]
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.early_stopping:
                print("Early stopping!")
                break

        # Adjust LR
        if args.lr_scheduler == 'ExponentialLR':
            scheduler.step()
        elif args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(logs['val']['loss'][-1])

    if epochs_since_best < args.early_stopping:
        print("Reached maximum number of epochs!")

    # save logs
    logs_path = os.path.join(args.base_dir, args.mtype, args.save_string, "logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(logs_path + f"/logs_{k}.pkl", "wb") as f:
        pickle.dump(logs, f)

    return logs
