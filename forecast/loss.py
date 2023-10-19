import numpy as np
import torch
from torch import nn
from pytorch_forecasting.metrics import MultiHorizonMetric
from sklearn.utils.class_weight import compute_class_weight

from utils.calc import percent_in_areas, get_peg_info


#################   NLL   #################

class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()
        self.criterion = nn.GaussianNLLLoss()

    def forward(self, output, target):
        return self.criterion(output[0], target[:, :, 0], output[1])


class NLLMetric(nn.Module):
    def __init__(self):
        super(NLLMetric, self).__init__()
        self.criterion = NLL()

    def forward(self, output, target):
        return self.criterion(output, target).item()


class PointwiseNLL(nn.Module):
    def __init__(self):
        super(PointwiseNLL, self).__init__()
        self.nll = nn.GaussianNLLLoss(reduction='none')

    def forward(self, output, target):
        pointwise_nll = torch.mean(self.nll(output[0], target[:, :, 0], output[1]), dim=0)
        return pointwise_nll.detach().cpu().numpy()


class LastNLL(nn.Module):
    def __init__(self):
        super(LastNLL, self).__init__()
        self.pointwise_nll = PointwiseNLL()

    def forward(self, output, target):
        return self.pointwise_nll(output, target)[-1].item()


#################   PEG   #################

# noinspection PyTypeChecker
class PEGLossElementwise(nn.Module):
    def __init__(self):
        super(PEGLossElementwise, self).__init__()
        self.s_A = 0.1
        self.s_B = 1.
        self.s_C = 2.
        self.s_D = 3.
        self.s_E = 4.

        self.peg_x_coords, self.peg_lines = get_peg_info()
        self.device = None
        self.zeros = None

    def calc_boundary_B(self, p, x):
        return torch.where(
            p,
            torch.where(x <= self.peg_x_coords[2],
                        torch.tensor(2.7746).to(self.device),
                        torch.where(x <= self.peg_x_coords[9],
                                    x * self.peg_lines['b'][1][1]['a'] + self.peg_lines['b'][1][1]['b'],
                                    torch.where(x <= self.peg_x_coords[13],
                                                x * self.peg_lines['b'][1][2]['a'] + self.peg_lines['b'][1][2]['b'],
                                                x * self.peg_lines['b'][1][3]['a'] + self.peg_lines['b'][1][3]['b']))),
            torch.where(x <= self.peg_x_coords[4],
                          self.zeros,
                          torch.where(x <= self.peg_x_coords[10],
                                      x * self.peg_lines['b'][-1][1]['a'] + self.peg_lines['b'][-1][1]['b'],
                                      torch.where(x <= self.peg_x_coords[14],
                                                  x * self.peg_lines['b'][-1][2]['a'] + self.peg_lines['b'][-1][2][
                                                      'b'],
                                                  x * self.peg_lines['b'][-1][3]['a'] + self.peg_lines['b'][-1][3][
                                                      'b']))))

    def calc_boundary_C(self, mask, p, x):
        return torch.where(
            mask,
            torch.where(p,
                          torch.where(x <= self.peg_x_coords[2],
                                      torch.tensor(3.3296).to(self.device),
                                      torch.where(x <= self.peg_x_coords[4],
                                                  x * self.peg_lines['c'][1][1]['a'] + self.peg_lines['c'][1][1]['b'],
                                                  torch.where(x <= self.peg_x_coords[5],
                                                              x * self.peg_lines['c'][1][2]['a'] + self.peg_lines['c'][1][2]['b'],
                                                              x * self.peg_lines['c'][1][3]['a'] + self.peg_lines['c'][1][3]['b']))),
                          torch.where(x <= self.peg_x_coords[7],
                                      torch.tensor(0).float().to(self.device),
                                      torch.where(x <= self.peg_x_coords[12],
                                                  x * self.peg_lines['c'][-1][1]['a'] + self.peg_lines['c'][-1][1]['b'],
                                                  x * self.peg_lines['c'][-1][2]['a'] + self.peg_lines['c'][-1][2]['b']))),
            self.zeros)

    def calc_boundary_D(self, mask, p, x):
        return torch.where(
            mask,
            torch.where(p,
                        torch.where(x <= self.peg_x_coords[1],
                                    torch.tensor(5.5493).to(self.device),
                                    torch.where(x <= self.peg_x_coords[4],
                                                x * self.peg_lines['d'][1][1]['a'] + self.peg_lines['d'][1][1]['b'],
                                                torch.where(x <= self.peg_x_coords[6],
                                                            x * self.peg_lines['d'][1][2]['a'] + self.peg_lines['d'][1][2]['b'],
                                                            x * self.peg_lines['d'][1][3]['a'] + self.peg_lines['d'][1][3]['b']))),
                        torch.where(x <= self.peg_x_coords[11],
                                    self.zeros,
                                    x * self.peg_lines['d'][-1][1]['a'] + self.peg_lines['d'][-1][1]['b'])),
            self.zeros)

    def calc_boundary_E(self, mask, p, x):
        return torch.where(
            mask,
            torch.where(p,
                        torch.where(x <= self.peg_x_coords[3],
                                    x * self.peg_lines['e'][1][0]['a'] + self.peg_lines['e'][1][0]['b'],
                                    x * self.peg_lines['e'][1][1]['a'] + self.peg_lines['e'][1][1]['b']),
                        self.zeros),
            self.zeros)

    def forward(self, output, target):
        x, y = target[:, :, 0], output[0]
        self.device = x.device
        self.zeros = torch.zeros_like(x)
        dist_yx = torch.abs(y - x)
        y_minus_x = y - x
        p = y_minus_x >= 0
        S = torch.zeros_like(x)

        # Distance in A of points from A^c
        b_x = self.calc_boundary_B(p, x)
        mask_area = dist_yx <= torch.abs(b_x - x)
        S[mask_area] = self.s_A
        mask_outer_areas = torch.logical_not(mask_area)
        dist_AB = torch.where(mask_outer_areas, torch.abs(b_x - x), self.zeros)

        # Distance in B of points from (A+B)^c
        c_x = self.calc_boundary_C(mask_outer_areas, p, x)
        mask_area = torch.clone(mask_outer_areas)
        mask_area[mask_outer_areas] = mask_area[mask_outer_areas] & (dist_yx[mask_outer_areas] <= torch.abs(c_x[mask_outer_areas] - x[mask_outer_areas]))
        S[mask_area] = self.s_B
        mask_outer_areas = mask_outer_areas & torch.logical_not(mask_area)
        dist_BC = torch.where(mask_outer_areas, torch.abs(c_x - b_x), self.zeros)

        # Distance in C of points from (A+B+C)^c
        d_x = self.calc_boundary_D(mask_outer_areas, p, x)
        mask_area = torch.clone(mask_outer_areas)
        mask_area[mask_outer_areas] = mask_area[mask_outer_areas] & (dist_yx[mask_outer_areas] <= torch.abs(d_x[mask_outer_areas] - x[mask_outer_areas]))
        S[mask_area] = self.s_C
        mask_outer_areas = mask_outer_areas & torch.logical_not(mask_area)
        dist_CD = torch.where(mask_outer_areas, torch.abs(d_x - c_x), self.zeros)

        # Distance in D of points from (A+B+C+D)^c
        e_x = self.calc_boundary_E(mask_outer_areas, p, x)
        mask_area = torch.clone(mask_outer_areas)
        mask_area[mask_outer_areas] = mask_area[mask_outer_areas] & (dist_yx[mask_outer_areas] <= torch.abs(e_x[mask_outer_areas] - x[mask_outer_areas]))
        S[mask_area] = self.s_D
        mask_outer_areas = mask_outer_areas & torch.logical_not(mask_area)
        dist_DE = torch.where(mask_outer_areas, torch.abs(e_x - d_x), self.zeros)

        # Slope in E of points from (A+B+C+D)^c
        S[mask_outer_areas] = self.s_E

        return S * dist_yx - ((S - self.s_A) * dist_AB + (S - self.s_B) * dist_BC + (S - self.s_C) * dist_CD + (S - self.s_D) * dist_DE)


class PEGLoss(nn.Module):
    def __init__(self):
        super(PEGLoss, self).__init__()
        self.peg_loss_elementwise = PEGLossElementwise()

    def forward(self, output, target):
        peg_loss_elementwise = self.peg_loss_elementwise(output, target)
        peg_loss = torch.mean(peg_loss_elementwise)
        return peg_loss


class PEGMetric(nn.Module):
    def __init__(self):
        super(PEGMetric, self).__init__()
        self.criterion = PEGLoss()

    def forward(self, output, target):
        return self.criterion(output, target).item()


class PointwisePEG(nn.Module):
    def __init__(self):
        super(PointwisePEG, self).__init__()
        self.peg_loss_elementwise = PEGLossElementwise()

    def forward(self, output, target):
        peg_loss_elementwise = self.peg_loss_elementwise(output, target)
        pointwise_peg = torch.mean(peg_loss_elementwise, dim=0)
        return pointwise_peg.detach().cpu().numpy()


class LastPEG(nn.Module):
    def __init__(self):
        super(LastPEG, self).__init__()
        self.pointwise_peg = PointwisePEG()

    def forward(self, output, target):
        return self.pointwise_peg(output, target)[-1].item()


#################   NLLPEG   #################

class NLLPEGSurface(nn.Module):
    def __init__(self, PEG_weight):
        super(NLLPEGSurface, self).__init__()
        self.criterion = nn.GaussianNLLLoss()
        self.peg_loss = PEGLoss()
        self.PEG_weight = PEG_weight

    def forward(self, output, target):
        nll = self.criterion(output[0], target[:, :, 0], output[1])
        peg = self.peg_loss(output, target)
        loss = nll + self.PEG_weight * peg
        return loss


class NLLPEGMetric(nn.Module):
    def __init__(self, PEG_weight):
        super(NLLPEGMetric, self).__init__()
        self.criterion = NLLPEGSurface(PEG_weight)

    def forward(self, output, target):
        return self.criterion(output, target).item()


class PointwiseNLLPEG(nn.Module):
    def __init__(self, PEG_weight):
        super(PointwiseNLLPEG, self).__init__()
        self.pointwise_nll = PointwiseNLL()
        self.pointwise_peg = PointwisePEG()
        self.PEG_weight = PEG_weight

    def forward(self, output, target):
        return self.pointwise_nll(output, target) + self.PEG_weight * self.pointwise_peg(output, target)


class LastNLLPEG(nn.Module):
    def __init__(self, PEG_weight):
        super(LastNLLPEG, self).__init__()
        self.pointwise_nllpeg = PointwiseNLLPEG(PEG_weight)

    def forward(self, output, target):
        return self.pointwise_nllpeg(output, target)[-1].item()


#################   RMSE   #################

class RMSEMetric(nn.Module):
    def __init__(self):
        super(RMSEMetric, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.mse(output[0], target[:, :, 0])).item()


class PointwiseRMSE(nn.Module):
    def __init__(self):
        super(PointwiseRMSE, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        pointwise_rmse = torch.sqrt(torch.mean(self.mse(output[0], target[:, :, 0]), dim=0))
        return pointwise_rmse.detach().cpu().numpy()


class LastRMSE(nn.Module):
    def __init__(self):
        super(LastRMSE, self).__init__()
        self.pointwise_rmse = PointwiseRMSE()

    def forward(self, output, target):
        return self.pointwise_rmse(output, target)[-1].item()


#################   MAE   #################

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, output, target):
        return self.mae(output[0], target[:, :, 0]).item()


class PointwiseMAE(nn.Module):
    def __init__(self):
        super(PointwiseMAE, self).__init__()
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        pointwise_mae = torch.mean(self.mae(output[0], target[:, :, 0]), dim=0)
        return pointwise_mae.detach().cpu().numpy()


#################   PEG Percentages   #################

class PEGPercentages(nn.Module):
    def __init__(self):
        super(PEGPercentages, self).__init__()
        self.calc_area_percentages = percent_in_areas

    def forward(self, output, target):
        return self.calc_area_percentages(output[0], target[:, :, 0], grid_type='parkes')


class PointwisePEGPercentages(nn.Module):
    def __init__(self):
        super(PointwisePEGPercentages, self).__init__()
        self.calc_area_percentages = percent_in_areas

    def forward(self, output, target):
        return self.calc_area_percentages(output[0], target[:, :, 0], grid_type='parkes', pointwise=True)


class LastPEGPercentages(nn.Module):
    def __init__(self):
        super(LastPEGPercentages, self).__init__()
        self.pointwise_area_percentages = PointwisePEGPercentages()

    def forward(self, output, target):
        return self.pointwise_area_percentages(output, target)[-1]


#################   CEG Percentages   #################

class CEGPercentages(nn.Module):
    def __init__(self):
        super(CEGPercentages, self).__init__()
        self.calc_area_percentages = percent_in_areas

    def forward(self, output, target):
        return self.calc_area_percentages(output[0], target[:, :, 0], grid_type='clarke')


class PointwiseCEGPercentages(nn.Module):
    def __init__(self):
        super(PointwiseCEGPercentages, self).__init__()
        self.calc_area_percentages = percent_in_areas

    def forward(self, output, target):
        return self.calc_area_percentages(output[0], target[:, :, 0], grid_type='clarke', pointwise=True)


class LastCEGPercentages(nn.Module):
    def __init__(self):
        super(LastCEGPercentages, self).__init__()
        self.pointwise_area_percentages = PointwiseCEGPercentages()

    def forward(self, output, target):
        return self.pointwise_area_percentages(output, target)[-1]


#################   Layer Loss   #################

def get_pos_weight(y):
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y), y=y.reshape(-1)), dtype=torch.float)
    return class_weights[1] / class_weights[0]


class LayerLoss(nn.Module):
    def __init__(
            self,
            device,
            criterion,
            sigma,
            lambda_OLS,
            lambda_orth,
            lambda_sparse,
            orthogonal_loss,
            sparse_coding
    ):
        super(LayerLoss, self).__init__()
        self.device = device
        self.sigma = sigma
        self.criterion = criterion
        self.lambda_OLS = lambda_OLS
        self.lambda_orth = lambda_orth
        self.lambda_sparse = lambda_sparse
        self.orthogonal_loss = orthogonal_loss
        self.sparse_coding = sparse_coding

    def forward(self, output, target, model):
        domain_bases = {i: model.gdu_layer.gdus[f'GDU_{i}'].domain_basis for i in range(model.num_gdus)}
        weight_matrix = model.gdu_layer.betas
        K = torch.zeros(len(domain_bases), len(domain_bases)).to(self.device)

        # Base Loss
        base_loss = self.criterion(output, target)

        # OLS Loss (3 terms)
        ols_term_1 = model.gdu_layer.k_x_x.mean()

        k_x_V = torch.stack([model.gdu_layer.gdus[f'GDU_{i}'].k_x_V_mean for i in range(model.num_gdus)], dim=1)
        ols_term_2_matrix = weight_matrix * k_x_V
        ols_term_2_vector = -2 * ols_term_2_matrix.sum(dim=1)
        ols_term_2 = ols_term_2_vector.mean()

        for domain_num_1, basis_matrix_1 in domain_bases.items():
            for domain_num_2, basis_matrix_2 in domain_bases.items():
                if domain_num_1 == domain_num_2:
                    K[domain_num_1, domain_num_2] = model.gdu_layer.gdus[f'GDU_{domain_num_1}'].k_V_V_mean
                elif domain_num_1 < domain_num_2:
                    K[domain_num_1, domain_num_2] = self.rbf_kernel(torch.permute(basis_matrix_1, (2, 1, 0)), basis_matrix_2).mean()
                elif domain_num_1 > domain_num_2:
                    K[domain_num_1, domain_num_2] = K[domain_num_2, domain_num_1]
        ols_term_3 = (weight_matrix @ K @ weight_matrix.transpose(0, 1)).diagonal().mean()
        ols = ols_term_1 + ols_term_2 + ols_term_3
        loss = base_loss + self.lambda_OLS * ols

        # Orthogonal Loss
        if self.orthogonal_loss:
            orth = torch.linalg.svdvals(K - K.diag().diag())[0] # largest singular value
            loss = loss + self.lambda_orth * orth

        # Sparse Coding Loss (L1 Norm of betas)
        if self.sparse_coding:
            sparse = weight_matrix.abs().sum(dim=1).mean(dim=0)
            loss = loss + self.lambda_sparse * sparse

        return loss

    def rbf_kernel(self, x, y):
        l2_dist = torch.sum(torch.square(x-y), dim=1)
        k_x_y = torch.exp(torch.mul(l2_dist, -1/(2*self.sigma**2)))
        return k_x_y


#################   Pytorch Forecasting NLL   #################

class NLLPytorchForecasting(MultiHorizonMetric):
    def __init__(self):
        super(NLLPytorchForecasting, self).__init__()
        self.criterion = nn.GaussianNLLLoss()

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor):
        loss = self.criterion(y_pred[0], target[:, :, 0], y_pred[1])
        return loss

