import torch
import torch.nn as nn


class GDULayer(nn.Module):
    def __init__(
            self,
            task,
            feature_vector_size,
            output_size,
            num_gdus=2,
            domain_dim=3,
            sigma=0.5,
            similarity_measure_name='MMD',
            softness_param=1
    ):
        super().__init__()
        self.task = task
        self.feature_vector_size = feature_vector_size
        self.output_size = output_size
        self.num_gdus = num_gdus
        self.domain_dim = domain_dim
        self.sigma = sigma
        self.similarity_measure_name = similarity_measure_name
        self.softness_param = softness_param

        # initialize M GDUs and learning machines
        self.gdus = nn.ModuleDict({
            f'GDU_{i}': GDU(
                i,
                self.domain_dim,
                self.feature_vector_size,
                self.sigma,
                self.similarity_measure_name,
            )
            for i in range(self.num_gdus)
        })
        self.learning_machines = nn.ModuleDict({
            f'learning_machine_{i}': LearningMachine(self.feature_vector_size,
                                                     self.output_size,
                                                     task=self.task)
            for i in range(self.num_gdus)
        })
        self.kernel_softmax = torch.nn.Softmax(dim=1)
        self.output_dim = 1 if task == 'classification' else 2
        self.betas = None
        self.k_x_x = None

    def forward(self, x_tilde):
        x = torch.unsqueeze(x_tilde, -1)
        self.k_x_x = self.rbf_kernel(x, x)
        self.betas = torch.stack([self.gdus[f'GDU_{i}'](x_tilde, self.k_x_x) for i in range(self.num_gdus)], dim=1)
        if self.similarity_measure_name in ['MMD', 'CS']:
            self.betas = self.kernel_softmax(self.softness_param * self.betas)
        y_tildes = torch.stack([self.learning_machines[f'learning_machine_{i}'](x_tilde) for i in range(self.num_gdus)], dim=1)
        return torch.sum(self.betas.unsqueeze(2).unsqueeze(3) * y_tildes, dim=1)

    def rbf_kernel(self, x, y):
        l2_dist = torch.sum(torch.square(x-y), dim=1)
        k_x_y = torch.exp(torch.mul(l2_dist, -1/(2*self.sigma**2)))
        return k_x_y


class GDU(nn.Module):
    def __init__(
            self,
            gdu_num,
            domain_dim,
            feature_vector_size,
            sigma,
            similarity_measure_name,
    ):
        super().__init__()
        self.gdu_num = gdu_num
        self.domain_dim = domain_dim
        self.feature_vector_size = feature_vector_size
        self.sigma = sigma
        self.similarity_measure_name = similarity_measure_name

        domain_basis_tensor = torch.normal(
            mean=torch.mul(torch.ones(self.feature_vector_size, self.domain_dim), self.gdu_num*0.5*(-1)**self.gdu_num),
            std=torch.mul(torch.ones(self.feature_vector_size, self.domain_dim), (self.gdu_num+1)*0.1)
        )
        domain_basis_tensor_batch_compatible = torch.unsqueeze(domain_basis_tensor, 0)
        self.domain_basis = torch.nn.Parameter(domain_basis_tensor_batch_compatible)

        self.k_x_V_mean = None
        self.k_V_V = None
        self.k_V_V_mean = None

    def forward(self, x, k_x_x):
        x = torch.unsqueeze(x, -1)
        self.k_x_V_mean = self.rbf_kernel(x, self.domain_basis).mean(dim=1)
        self.k_V_V = self.rbf_kernel(torch.permute(self.domain_basis, (2, 1, 0)), self.domain_basis)
        self.k_V_V_mean = self.k_V_V.mean()

        if self.similarity_measure_name == 'MMD':
            k_x_x = torch.squeeze(k_x_x)
            beta = -(k_x_x - 2 * self.k_x_V_mean + self.k_V_V_mean)
        elif self.similarity_measure_name == 'CS':
            k_x_x = torch.squeeze(k_x_x)
            beta = self.k_x_V_mean / (k_x_x.sqrt() * self.k_V_V.sqrt().mean())
        elif self.similarity_measure_name == 'Projected':
            beta = self.k_x_V_mean / self.k_V_V_mean
        return beta

    def rbf_kernel(self, x, y):
        l2_dist = torch.sum(torch.square(x-y), dim=1)
        k_x_y = torch.exp(l2_dist * -1/(2*self.sigma**2))
        return k_x_y


class LearningMachine(nn.Module):
    def __init__(self, feature_vector_size, output_size, task):
        super().__init__()
        self.task = task

        if self.task == 'classification':
            self.linear = nn.Linear(feature_vector_size, output_size)
            self.activation = nn.Tanh()
        elif self.task == 'probabilistic_forecasting':
            self.linear_mu = nn.Linear(feature_vector_size, output_size)
            self.linear_sigma = nn.Linear(feature_vector_size, output_size)

    def forward(self, x_tilde):
        if self.task == 'classification':
            x = self.linear(x_tilde)
            y_tilde = self.activation(x)
            y_tilde = y_tilde.unsqueeze(dim=2)
        elif self.task == 'probabilistic_forecasting':
            mu = self.linear_mu(x_tilde)
            sigma = torch.exp(self.linear_sigma(x_tilde))
            y_tilde = torch.stack([mu, sigma], dim=2)
        return y_tilde
