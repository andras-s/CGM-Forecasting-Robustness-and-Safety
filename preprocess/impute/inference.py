import torch
import gpytorch
import warnings
from utils.plot import plot_gaussian_process


def gp_sampling(inputs, target, model, sample_length=24, sample_step=5, inputs_gp=None, plot_style='white', save_plot=None):
    """
    uses a pretrained Gaussian process model and infers equally spaced samples from unequally spaced input data

    Parameters
    ----------
    inputs : torch.tensor
        the inputs for which the sampling should be conducted
    targets : torch.tensor
        the targets for which the sampling should be conducted
    model : torch model
        pretrained model
    sample_length : float
        the time length for which the inference should be conducted (in hours)
    sample_step : int, optional
        the time between sampled points (in minutes)
    save_plot : string, optional
        whether the Gaussian process posterior should be plotted

    Returns
    -------
    sampled_mean : torch.Tensor
        the mean values for the desired points in time
    sampled_var : torch.Tensor (only if return_var=True)
        the estimated variance for the desired points in time
    """
    device = model.device
    inputs, target = inputs.to(device), target.to(device)
    model.set_train_data(inputs, target, strict=False)

    if inputs_gp is None:
        inputs_gp = torch.arange(start=int(inputs[0].item()), end=int(inputs[0].item())+int(sample_length * 60), step=sample_step, dtype=torch.float32).to(device)
    else:
        inputs_gp = inputs_gp.to(device)

    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output_gp = model(inputs_gp)

    if save_plot is not None:
        plot_gaussian_process(inputs, target, inputs_gp, output_gp, style=plot_style, save_to=save_plot)

    warnings.simplefilter('ignore', UserWarning)
    sampled_mean = torch.tensor(output_gp.mean.detach().clone(), dtype=torch.float32).cpu()
    sampled_var = torch.tensor(output_gp.variance.detach().clone(), dtype=torch.float32).cpu()

    return sampled_mean, sampled_var
