import torch
import gpytorch


class GaussianProcess(gpytorch.models.ExactGP):
    """Gaussian process model class"""
    def __init__(self, train_x, train_y, likelihood, args):
        super(GaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = self.get_mean_module(args.mean)
        self.mean_module.constant = torch.nn.Parameter(torch.Tensor([args.initial_mean]))
        if 'SM' not in args.kernel:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    self.get_kernel(args.kernel, args.num_mixtures), grid_size=args.grid_size, num_dims=1))
        elif 'SM' in args.kernel:
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixtures)
            if train_x is not None and train_y is not None:
                self.covar_module.initialize_from_data(train_x, train_y)

            if args.frequency_upper_limit != 0:
                self.covar_module.register_constraint('raw_mixture_means', gpytorch.constraints.Interval(args.frequency_lower_limit, args.frequency_upper_limit))
            if args.decay_upper_limit != 0:
                self.covar_module.register_constraint('raw_mixture_scales', gpytorch.constraints.Interval(0., args.decay_upper_limit))

            if len(args.initial_frequencies) > 0:
                self.covar_module.mixture_means = torch.tensor(args.initial_frequencies)
                self.covar_module.mixture_weights = torch.tensor(1 / args.num_mixtures)
                if args.freeze_frequencies:
                    self.covar_module.raw_mixture_means.requires_grad_(False)
            if len(args.initial_decays) > 0:
                self.covar_module.mixture_scales = torch.tensor(args.initial_decays)
                self.covar_module.mixture_weights = torch.tensor(1 / args.num_mixtures)
                if args.freeze_decays:
                    self.covar_module.raw_mixture_scales.requires_grad_(False)

    def get_mean_module(self, mean):
        """Helper function to fetch a kernel function"""
        if "zero" in mean:
            return gpytorch.means.ZeroMean()
        elif "constant" in mean:
            return gpytorch.means.ConstantMean()
        elif "linear" in mean:
            return gpytorch.means.LinearMean(input_size=1)
        else:
            raise NotImplementedError

    def get_kernel(self, kernel, num_mixtures):
        """Helper function to fetch a kernel function"""
        if "RBF" in kernel:
            return gpytorch.kernels.RBFKernel()
        elif "linear" in kernel:
            return gpytorch.kernels.LinearKernel()
        elif "Matern-12" in kernel:
            return gpytorch.kernels.MaternKernel(nu=1/2)
        elif "Matern-32" in kernel:
            return gpytorch.kernels.MaternKernel(nu=3/2)
        elif "Matern-52" in kernel:
            return gpytorch.kernels.MaternKernel(nu=5/2)
        elif "Cosine" in kernel:
            return gpytorch.kernels.CosineKernel()
        elif "SM" in kernel:
            return gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        else:
            raise NotImplementedError

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
