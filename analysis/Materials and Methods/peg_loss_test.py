import torch

from utils.calc import get_peg_info
from utils.plot import visualize_loss_surface
from forecast.loss import PEGLossElementwise



device = torch.device('cpu')

s_a = 0.1
s_b = 1.
s_c = 2.
s_d = 3.
s_e = 4.

peg_x_coords, peg_lines = get_peg_info()


x = torch.linspace(0, 25, steps=2501)
y = torch.linspace(0, 25, steps=2501)
grid_x, grid_y = torch.meshgrid(x, y)


# Test the loss class
peg_loss_instance = PEGLossElementwise()

grid_x_2, grid_y_2 = torch.unsqueeze(grid_x, 2).to(device), torch.unsqueeze(grid_y, 0).to(device)
# grid_x_2[:, :, 0] should be 1024, 13 (in train)
# grid_y_2[0]       should be 1024, 13 (in train)
loss_2 = peg_loss_instance(grid_y_2, grid_x_2)
visualize_loss_surface(grid_x, grid_y, loss_2.cpu())

