import os
import torch
import numpy as np
from utils.calc import calculate_peg_loss, calculate_peg_derivative
from utils.plot import plot_error_grid, plot_peg_loss

version = '2'
device = torch.device('cpu')
save_dir = '/analysis/results/error_grids'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# output, target = torch.tensor([0]), torch.tensor([0])
# plot_error_grid(output, target, domain_name='', grid_type='clarke', units='mmoll', save_to=save_dir + '/ceg', plot_show=True)
# plot_error_grid(output, target, domain_name='', grid_type='parkes', units='mmoll', save_to=save_dir + '/peg', plot_show=True)

X = np.arange(0, 25, 0.01)
Y = np.arange(0, 25, 0.01)
X, Y = np.meshgrid(X, Y)

L = calculate_peg_loss(X, Y, device=device)
L_y = calculate_peg_derivative(X, Y)
plot_peg_loss(X, Y, L, L_y, save_dir=save_dir, version=version)


