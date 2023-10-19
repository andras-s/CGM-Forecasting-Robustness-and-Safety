import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, Baseline
from forecast.loss import NLLPytorchForecasting
from forecast.log_sparse_transfomer.log_sparse_transformer import TransformerModel
from forecast.gdu_layer.gdu import GDULayer, LearningMachine


class t_0(nn.Module):
    """Baseline model used for forecasting"""
    def __init__(self, device, train_data):
        super().__init__()
        self.device = device
        self.horizon = train_data.target.size(1)
        self.sigmas = torch.Tensor()
        self.calc_sigmas(train_data)

    def forward(self, x):
        return x[:, -1].repeat(1, self.horizon), self.sigmas.repeat(x.size(0), 1)

    def calc_sigmas(self, train_data):
        targets = train_data.target.squeeze()
        outputs = self.forward(train_data.inputs)[0]
        self.sigmas = torch.sqrt(torch.sum(torch.square(outputs - targets), dim=0) / targets.size(0))


class RegularLSTM(nn.Module):
    """Class of the predictive model used for forecasting"""
    def __init__(self, device, num_input_features, hidden_size, num_layers, lin_layers=(512, 256), dropouts=(0.2, 0.3)):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lin_layer_input_sizes = [hidden_size] + [inp_size for inp_size in lin_layers[:-1]]
        lin_layer_output_sizes = [inp_size for inp_size in lin_layers]

        self.lstm = nn.LSTM(input_size=num_input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        linear_layers = [
            nn.Sequential(
                nn.Linear(lin_layer_input_sizes[i], lin_layer_output_sizes[i]),
                nn.Dropout(p=dropouts[i]),
                nn.ReLU()
            ) for i in range(len(lin_layers))
        ]
        self.linear = nn.Sequential(*linear_layers)

        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_size), torch.zeros(num_layers, 1, self.hidden_size))

    def init_hidden_cell(self, x):
        # reset the LSTM cell
        return torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device), \
               torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

    def forward(self, x):
        self.hidden_cell = self.init_hidden_cell(x)

        # flattening the weights when on GPU to use less memory
        if self.device.type == 'cuda':
            self.lstm.flatten_parameters()

        x, self.hidden_cell = self.lstm(x, self.hidden_cell)
        return self.linear(x[:, -1, :])


class ConvTransformerFE(nn.Module):
    """Class of the predictive model used for forecasting"""
    def __init__(self,
                 device,
                 n_time_series,
                 forecast_history,
                 sub_len,
                 q_len,
                 n_embd,
                 n_head,
                 num_layer,
                 scale_att,
                 dropout,
                 lin_layers,
                 lin_dropouts,
                 additional_params
                 ):
        super().__init__()

        self.conv_transformer = TransformerModel(
            device=device,
            n_time_series=n_time_series,
            forecast_history=forecast_history,
            sub_len=sub_len,
            q_len=q_len,
            n_embd=n_embd,
            n_head=n_head,
            num_layer=num_layer,
            scale_att=scale_att,
            dropout=dropout,
            additional_params=additional_params
        )

        lin_layer_input_sizes = [(n_time_series + n_embd) * forecast_history] + [inp_size for inp_size in lin_layers[:-1]]
        lin_layer_output_sizes = [inp_size for inp_size in lin_layers]
        linear_layers = [
            nn.Sequential(
                nn.Linear(lin_layer_input_sizes[i], lin_layer_output_sizes[i]),
                nn.Dropout(p=lin_dropouts[i]),
                nn.ReLU()
            ) for i in range(len(lin_layers))
        ]
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv_transformer(series_id=None, x=x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class RegressionLSTM(nn.Module):
    def __init__(
            self,
            device,
            num_input_features,
            hidden_size,
            num_layers,
            output_size,
            lin_layers=(512, 256),
            dropouts=(0.2, 0.3)
    ):
        super().__init__()

        self.lstm = RegularLSTM(device, num_input_features, hidden_size, num_layers, lin_layers, dropouts)

        self.linear_mu = nn.Linear(lin_layers[-1], output_size)
        self.linear_sigma = nn.Linear(lin_layers[-1], output_size)

    def forward(self, x):
        x = self.lstm(x)

        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))

        return mu, sigma


class RegressionConvTransformer(nn.Module):
    def __init__(
            self,
            device,
            n_time_series=1,
            forecast_history=72,
            output_size=7,
            sub_len=None,
            q_len=4,
            n_embd=20,
            n_head=8,
            num_layer=3,
            scale_att=True,
            dropout=0.1,
            lin_layers=(512, 256),
            lin_dropouts=(0.2, 0.3),
            additional_params=None
    ):
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            output_size: The number of targets to forecast
            sub_len: sub_len of the sparse attention
            num_layer: The number of transformer blocks in the model
            n_embd: The dimension of Position embedding and time series ID embedding
            forecast_history: The number of historical steps fed into the time series model
            dropout: The dropout for the embedding of the model
            additional_params: Additional parameters used to initialize the attention model. Can include sparse=True/False
        """
        super(RegressionConvTransformer, self).__init__()
        self.conv_transformer_fe = ConvTransformerFE(
            device,
            n_time_series,
            forecast_history,
            sub_len,
            q_len,
            n_embd,
            n_head,
            num_layer,
            scale_att,
            dropout,
            lin_layers,
            lin_dropouts,
            additional_params
        )
        self.linear_mu = nn.Linear(lin_layers[-1], output_size)
        self.linear_sigma = nn.Linear(lin_layers[-1], output_size)
        self.softplus = nn.Softplus()
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_transformer_fe(x)
        mu = self.linear_mu(x)
        sigma = self.softplus(self.linear_sigma(x))
        return mu, sigma

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class RegressionTFT(nn.Module):
    def __init__(
            self,
            hidden_size,
            attention_head_size,
            dropout,
            hidden_continuous_size,
            device,
            train_timeseries_dataset
    ):
        super().__init__()
        self.tft = TemporalFusionTransformer.from_dataset(
            train_timeseries_dataset,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=2,
            loss=NLLPytorchForecasting()
        )
        self.tft = self.tft.to(device)

    def forward(self, x):
        x = self.tft(x)
        mu = x.prediction[:, :, 0]
        sigma = torch.exp(x.prediction[:, :, 1])
        return mu, sigma


class EnsemblePredictor(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            output_size,
            num_heads
    ):
        super().__init__()
        self.feature_vector_size = feature_vector_size
        self.output_size = output_size
        self.num_heads = num_heads

        self.learning_machines = nn.ModuleDict({
            f'learning_machine_{i}': LearningMachine(self.feature_vector_size,
                                                     self.output_size,
                                                     task='probabilistic_forecasting')
            for i in range(self.num_heads)
        })

    def forward(self, x):
        x = [self.learning_machines[f'learning_machine_{i}'](x) for i in range(self.num_heads)]
        x = torch.mean(torch.stack(x, dim=3), dim=3)
        return x


class EnsembleModel(nn.Module):
    def __init__(
            self,
            feature_extractor,
            feature_vector_size,
            output_size,
            num_heads=10
    ):
        super().__init__()
        self.feature_vector_size = feature_vector_size
        self.output_size = output_size
        self.num_heads = num_heads

        self.feature_extractor = feature_extractor
        self.ensemble_predictor = EnsemblePredictor(
            feature_vector_size=feature_vector_size,
            output_size=output_size,
            num_heads=num_heads
        )

    def forward(self, x):               # x: (batch_size, input_len, 1)
        x = self.feature_extractor(x)   # x: (batch_size, hidden_size)
        x = self.ensemble_predictor(x)  # weighted_prediction: (batch_size, pred_len, 2)
        return x[:, :, 0], x[:, :, 1]


class LayerModel(nn.Module):
    def __init__(
            self,
            device,
            task,
            feature_extractor,
            feature_vector_size,
            output_size,
            num_gdus=5,
            domain_dim=10,
            sigma=2,
            similarity_measure_name='CS',
            softness_param=1
    ):
        super().__init__()
        self.device = device
        self.task = task
        self.feature_vector_size = feature_vector_size
        self.output_size = output_size
        self.num_gdus = num_gdus
        self.domain_dim = domain_dim
        self.sigma = sigma
        self.similarity_measure_name = similarity_measure_name
        self.softness_param = softness_param

        self.feature_extractor = feature_extractor
        self.gdu_layer = GDULayer(
            task=self.task,
            feature_vector_size=self.feature_vector_size,
            output_size=self.output_size,
            num_gdus=self.num_gdus,
            domain_dim=self.domain_dim,
            sigma=self.sigma,
            similarity_measure_name=self.similarity_measure_name,
            softness_param=self.softness_param
        )
        self.softmax = nn.Softmax(dim=1)
        self.x_tilde = None

    def forward(self, x):                                   # x: (batch_size, input_len, 1)
        self.x_tilde = self.feature_extractor(x)            # x_tilde: (batch_size, hidden_size)
        weighted_prediction = self.gdu_layer(self.x_tilde)  # weighted_prediction: (batch_size, pred_len, 1/2)
        if self.task == 'classification':
            output = weighted_prediction.squeeze()
            output = self.softmax(output)
        elif self.task == 'probabilistic_forecasting':
            mu = weighted_prediction[:, :, 0]               # mu: (batch_size, pred_len)
            sigma = weighted_prediction[:, :, 1]            # sigma: (batch_size, pred_len)
            output = mu, sigma
        return output


class RegressionBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline = Baseline()

    def forward(self, x):
        x = self.baseline(x)

        mu = x.prediction[:, :, 0]
        sigma = torch.exp(x.prediction[:, :, 1])

        return mu, sigma


class ClassificationLSTM(nn.Module):
    def __init__(self, device, num_input_features, hidden_size, num_layers, output_size=1, lin_layers=2):
        super().__init__()

        self.lstm = RegularLSTM(device, num_input_features, hidden_size, lin_layers)

        self.linear_mu = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.lstm(x)

        logits = self.linear_mu(x)

        return logits


