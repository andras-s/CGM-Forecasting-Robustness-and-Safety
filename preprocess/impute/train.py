import pandas as pd
import numpy as np
import torch
import gpytorch
import copy

from preprocess.impute.evaluation import calculate_batch_kpis, aggregate_kpis


def forward(model, criterion, inputs, target, lengths):
    """
    Calculate loss over entire batch
    """
    loss = torch.tensor([0], requires_grad=True, dtype=torch.float64).to(model.device)

    for i in range(len(inputs)):
        inp, tar = inputs[i][:lengths[i]], target[i][:lengths[i]]  # remove padding

        model.set_train_data(inp, tar, strict=False)
        try:
            outp = model(inp)
        except:
            print(f"Input is: {inp}")
            print(f"Target is: {tar}")
            pass
        try:
            loss = loss - criterion(outp, tar)
        except:
            pass
    loss = loss / len(inputs)

    return model, loss


def train_model(likelihood,
                model,
                kernel,
                criterion,
                optimiser,
                scheduler,
                metrics,
                train_loader,
                val_loader,
                args,
                k):

    # hp_names = [f"{('model.' + param_name).replace('raw_', '').split('.')[-1]}" for param_name, _ in model.named_parameters()]
    # logs_fold = pd.DataFrame(columns=['fold', 'epoch', 'batch', 'train_loss', 'val_loss', 'kernel'] + hp_names)
    kpi_columns = ['RMSE', 'mean_volatility', 'variance_volatility', 'variance_avg', 'num_variance_collapses']
    train_kpi_columns = ['train_' + kpi for kpi in kpi_columns]
    val_kpi_columns = ['val_' + kpi for kpi in kpi_columns]
    kpi_columns = ['KPI_score'] + train_kpi_columns + val_kpi_columns
    logs_columns = ['fold', 'epoch', 'batch', 'train_loss', 'val_loss'] + kpi_columns + ['kernel']
    logs_fold = pd.DataFrame(columns=logs_columns)
    logs_fold_kpi_entry = {kpi: np.nan for kpi in kpi_columns}

    min_loss = np.inf
    device = model.device

    for e in range(args.epochs):

        # Training
        model.eval()
        likelihood.train()
        batch_num = 0
        for (inputs, target, lengths) in train_loader:
            optimiser.zero_grad()

            inputs, target = inputs.to(device), target.to(device)
            model, loss = forward(model, criterion, inputs, target, lengths)
            # Create log entry from loss and hyperparameters
            logs_fold_entry = {"fold": k + 1, "epoch": e + 1, "batch": batch_num+1, "train_loss": loss.item(),
                               "val_loss": np.nan} | logs_fold_kpi_entry | {"kernel": kernel}
            for param_name, param in model.named_parameters():
                actual_param_name = ('model.' + param_name).replace("raw_", "")
                actual_param = eval(actual_param_name)
                if len(actual_param.size()) == 0:
                    logs_fold_entry[f"{actual_param_name.split('.')[-1]}"] = actual_param.item()
                elif len(actual_param) == 1:
                    logs_fold_entry[f"{actual_param_name.split('.')[-1]}"] = actual_param.item()
                else:
                    for param_num in range(len(actual_param)):
                        logs_fold_entry[f"{actual_param_name.split('.')[-1]}_{param_num+1}"] = actual_param[param_num].item()
            logs_fold_entry["learning_rate"] = optimiser.param_groups[0]["lr"]
            logs_fold = logs_fold.append(logs_fold_entry.copy(), ignore_index=True)
            print(logs_fold_entry)
            # Make training step

            loss.backward()
            optimiser.step()
            scheduler.step()
            batch_num += 1

        if args.noise_final_gp != 0:
            likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(0., args.noise_final_gp+1))
            likelihood.noise = args.noise_final_gp

        # Evaluation
        model.eval()
        likelihood.eval()

        train_batch_losses = []
        val_batch_losses = []
        train_batch_kpis = pd.DataFrame()
        val_batch_kpis = pd.DataFrame()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            ## on train data
            for (inputs, target, lengths) in train_loader:
                inputs, target = inputs.to(device), target.to(device)
                model, loss = forward(model, criterion, inputs, target, lengths)
                train_batch_losses.append(loss.item())

                aggr_batch_kpis = calculate_batch_kpis(inputs, target, lengths, model, sample_length=args.window_length, sample_step=5)
                train_batch_kpis = train_batch_kpis.append(aggr_batch_kpis.copy(), ignore_index=True)
            ## on validation data
            for (inputs, target, lengths) in val_loader:
                inputs, target = inputs.to(device), target.to(device)
                model, loss = forward(model, criterion, inputs, target, lengths)
                val_batch_losses.append(loss.item())

                aggr_batch_kpis = calculate_batch_kpis(inputs, target, lengths, model, sample_length=args.window_length, sample_step=5)
                val_batch_kpis = val_batch_kpis.append(aggr_batch_kpis.copy(), ignore_index=True)
        train_loss = sum(train_batch_losses) / len(train_batch_losses)
        val_loss = sum(val_batch_losses) / len(val_batch_losses)
        train_kpis = aggregate_kpis(train_batch_kpis, train_val='train')
        val_kpis = aggregate_kpis(val_batch_kpis, train_val='val')
        kpi_score = np.log(np.sqrt((train_kpis['train_RMSE']**2 + val_kpis['val_RMSE']**2)/2)) + np.log(1 + train_kpis['train_num_variance_collapses'] + val_kpis['val_num_variance_collapses'])

        ## Create log entry from loss and hyperparameters
        logs_fold_entry = {"fold": k + 1, "epoch": e + 1, "batch": np.nan, "train_loss": train_loss,
                           "val_loss": val_loss} | {'KPI_score': kpi_score} | train_kpis | val_kpis | {"kernel": kernel}
        for param_name, param in model.named_parameters():
            actual_param_name = ('model.' + param_name).replace("raw_", "")
            actual_param = eval(actual_param_name)
            if len(actual_param.size()) == 0:
                logs_fold_entry[f"{actual_param_name.split('.')[-1]}"] = actual_param.item()
            elif len(actual_param) == 1:
                logs_fold_entry[f"{actual_param_name.split('.')[-1]}"] = actual_param.item()
            else:
                for param_num in range(len(actual_param)):
                    logs_fold_entry[f"{actual_param_name.split('.')[-1]}_{param_num + 1}"] = actual_param[param_num].item()
        logs_fold_entry["learning_rate"] = np.nan
        logs_fold = logs_fold.append(logs_fold_entry.copy(), ignore_index=True)
        print(logs_fold_entry)

        # check if last step led to improved model and check early-stopping criterion
        if logs_fold['val_loss'].iloc[-1] < min_loss:
            min_loss = logs_fold['val_loss'].iloc[-1]
            best_model = copy.deepcopy(model)

        elif (np.array(logs_fold['val_loss'][-args.early_stopping:]) > min_loss).sum() == args.early_stopping:
            print("Early stopping!")
            return best_model, logs_fold

    return best_model, logs_fold
