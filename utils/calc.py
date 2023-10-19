import pandas as pd
import numpy as np
import torch
from copy import copy

# glucose levels in mg/dL
glucose_levels = {'hypo L2':    (0, 53),
                  'hypo L1':    (54, 69),
                  'target':     (70, 180),
                  'hyper L1':   (181, 250),
                  'hyper L2':   (251, 10000)}

MMOLL_MGDL = 18.02

SENSOR_INTERVAL = {'DEXCOM':    5,
                   'LIBRE':     15,
                   'MEDTRONIC': 5,
                   'MM670G':    5,
                   'MM640G':    5,
                   'FSL':       15}

lims = [0., 450.]
peg_points = np.array(
    # diagonal
    [[lims, lims],
     # B
     [[lims[0], 30], [50, 50]],  # B upper
     [[30, 140], [50, 170]],  # B upper
     [[140, 280], [170, 380]],  # B upper
     [[280, 430], [380, 550]],  # B upper
     [[50, 50], [lims[0], 30]],  # B lower
     [[50, 170], [30, 145]],  # B lower
     [[170, 385], [145, 300]],  # B lower
     [[385, 550], [300, 450]],  # B lower
     # C
     [[lims[0], 30], [60, 60]],  # C upper
     [[30, 50], [60, 80]],  # C upper
     [[50, 70], [80, 110]],  # C upper
     [[70, 260], [110, 550]],  # C upper

     [[120, 120], [lims[0], 30]],  # C lower
     [[120, 260], [30, 130]],  # C lower
     [[260, 550], [130, 250]],  # C lower
     # D
     [[lims[0], 25], [100, 100]],  # D upper
     [[25, 50], [100, 125]],  # D upper
     [[50, 80], [125, 215]],  # D upper
     [[80, 125], [215, 550]],  # D upper

     [[250, 250], [lims[0], 40]],  # D lower
     [[250, 550], [40, 150]],  # D lower
     # E
     [[lims[0], 35], [150, 155]],  # E upper
     [[35, 50], [155, 550]]])  # E upper

peg_points *= 1 / MMOLL_MGDL
peg_points = {
    'a': {1: {0: {'x': peg_points[0, 0], 'y':  peg_points[0, 1]}}},
    'b': {1: {0: {'x': peg_points[1, 0], 'y': peg_points[1, 1]},
              1: {'x': peg_points[2, 0], 'y': peg_points[2, 1]},
              2: {'x': peg_points[3, 0], 'y': peg_points[3, 1]},
              3: {'x': peg_points[4, 0], 'y': peg_points[4, 1]}},
          -1: {0: {'x': peg_points[5, 0], 'y': peg_points[5, 1]},
               1: {'x': peg_points[6, 0], 'y': peg_points[6, 1]},
               2: {'x': peg_points[7, 0], 'y': peg_points[7, 1]},
               3: {'x': peg_points[8, 0], 'y': peg_points[8, 1]}}},
    'c': {1: {0: {'x': peg_points[9, 0], 'y': peg_points[9, 1]},
              1: {'x': peg_points[10, 0], 'y': peg_points[10, 1]},
              2: {'x': peg_points[11, 0], 'y': peg_points[11, 1]},
              3: {'x': peg_points[12, 0], 'y': peg_points[12, 1]}},
          -1: {0: {'x': peg_points[13, 0], 'y': peg_points[13, 1]},
               1: {'x': peg_points[14, 0], 'y': peg_points[14, 1]},
               2: {'x': peg_points[15, 0], 'y': peg_points[15, 1]}}},
    'd': {1: {0: {'x': peg_points[16, 0], 'y': peg_points[16, 1]},
              1: {'x': peg_points[17, 0], 'y': peg_points[17, 1]},
              2: {'x': peg_points[18, 0], 'y': peg_points[18, 1]},
              3: {'x': peg_points[19, 0], 'y': peg_points[19, 1]}},
          -1: {0: {'x': peg_points[20, 0], 'y': peg_points[20, 1]},
               1: {'x': peg_points[21, 0], 'y': peg_points[21, 1]}}},
    'e': {1: {0: {'x': peg_points[22, 0], 'y': peg_points[22, 1]},
              1: {'x': peg_points[23, 0], 'y': peg_points[23, 1]}}}}

peg_x_coords = []
for area in peg_points.values():
    for sign in area.values():
        for line in sign.values():
            peg_x_coords.append(line['x'][0])
            peg_x_coords.append(line['x'][1])
peg_x_coords = list(set(peg_x_coords))
peg_x_coords.sort()

peg_lines = {}
for area_name, area in peg_points.items():
    peg_lines[area_name] = {}
    for sign_name, sign in area.items():
        peg_lines[area_name][sign_name] = {}
        for line_num, coords in sign.items():
            peg_lines[area_name][sign_name][line_num] = {}
            x1 = coords['x'][0]
            x2 = coords['x'][1]
            y1 = coords['y'][0]
            y2 = coords['y'][1]
            a = (y2 - y1) / (x2 - x1) if x1 != x2 else np.inf
            b = y1 - x1 * a
            peg_lines[area_name][sign_name][line_num]['a'] = a
            peg_lines[area_name][sign_name][line_num]['b'] = b


def get_peg_info():
    lims = [0., 450.]
    peg_points = np.array(
        # diagonal
        [[lims, lims],
         # B
         [[lims[0], 30], [50, 50]],  # B upper
         [[30, 140], [50, 170]],  # B upper
         [[140, 280], [170, 380]],  # B upper
         [[280, 430], [380, 550]],  # B upper
         [[50, 50], [lims[0], 30]],  # B lower
         [[50, 170], [30, 145]],  # B lower
         [[170, 385], [145, 300]],  # B lower
         [[385, 550], [300, 450]],  # B lower
         # C
         [[lims[0], 30], [60, 60]],  # C upper
         [[30, 50], [60, 80]],  # C upper
         [[50, 70], [80, 110]],  # C upper
         [[70, 260], [110, 550]],  # C upper

         [[120, 120], [lims[0], 30]],  # C lower
         [[120, 260], [30, 130]],  # C lower
         [[260, 550], [130, 250]],  # C lower
         # D
         [[lims[0], 25], [100, 100]],  # D upper
         [[25, 50], [100, 125]],  # D upper
         [[50, 80], [125, 215]],  # D upper
         [[80, 125], [215, 550]],  # D upper

         [[250, 250], [lims[0], 40]],  # D lower
         [[250, 550], [40, 150]],  # D lower
         # E
         [[lims[0], 35], [150, 155]],  # E upper
         [[35, 50], [155, 550]]])  # E upper

    peg_points *= 1 / MMOLL_MGDL
    peg_points = peg_points.astype(float)
    peg_points = {
        'a': {1: {0: {'x': peg_points[0, 0].tolist(), 'y': peg_points[0, 1].tolist()}}},
        'b': {1: {0: {'x': peg_points[1, 0].tolist(), 'y': peg_points[1, 1].tolist()},
                  1: {'x': peg_points[2, 0].tolist(), 'y': peg_points[2, 1].tolist()},
                  2: {'x': peg_points[3, 0].tolist(), 'y': peg_points[3, 1].tolist()},
                  3: {'x': peg_points[4, 0].tolist(), 'y': peg_points[4, 1].tolist()}},
              -1: {0: {'x': peg_points[5, 0].tolist(), 'y': peg_points[5, 1].tolist()},
                   1: {'x': peg_points[6, 0].tolist(), 'y': peg_points[6, 1].tolist()},
                   2: {'x': peg_points[7, 0].tolist(), 'y': peg_points[7, 1].tolist()},
                   3: {'x': peg_points[8, 0].tolist(), 'y': peg_points[8, 1].tolist()}}},
        'c': {1: {0: {'x': peg_points[9, 0].tolist(), 'y': peg_points[9, 1].tolist()},
                  1: {'x': peg_points[10, 0].tolist(), 'y': peg_points[10, 1].tolist()},
                  2: {'x': peg_points[11, 0].tolist(), 'y': peg_points[11, 1].tolist()},
                  3: {'x': peg_points[12, 0].tolist(), 'y': peg_points[12, 1].tolist()}},
              -1: {0: {'x': peg_points[13, 0].tolist(), 'y': peg_points[13, 1].tolist()},
                   1: {'x': peg_points[14, 0].tolist(), 'y': peg_points[14, 1].tolist()},
                   2: {'x': peg_points[15, 0].tolist(), 'y': peg_points[15, 1].tolist()}}},
        'd': {1: {0: {'x': peg_points[16, 0].tolist(), 'y': peg_points[16, 1].tolist()},
                  1: {'x': peg_points[17, 0].tolist(), 'y': peg_points[17, 1].tolist()},
                  2: {'x': peg_points[18, 0].tolist(), 'y': peg_points[18, 1].tolist()},
                  3: {'x': peg_points[19, 0].tolist(), 'y': peg_points[19, 1].tolist()}},
              -1: {0: {'x': peg_points[20, 0].tolist(), 'y': peg_points[20, 1].tolist()},
                   1: {'x': peg_points[21, 0].tolist(), 'y': peg_points[21, 1].tolist()}}},
        'e': {1: {0: {'x': peg_points[22, 0].tolist(), 'y': peg_points[22, 1].tolist()},
                  1: {'x': peg_points[23, 0].tolist(), 'y': peg_points[23, 1].tolist()}}}}

    peg_x_coords = []
    for area in peg_points.values():
        for sign in area.values():
            for line in sign.values():
                peg_x_coords.append(line['x'][0])
                peg_x_coords.append(line['x'][1])
    peg_x_coords = list(set(peg_x_coords))
    peg_x_coords.sort()

    peg_lines = {}
    for area_name, area in peg_points.items():
        peg_lines[area_name] = {}
        for sign_name, sign in area.items():
            peg_lines[area_name][sign_name] = {}
            for line_num, coords in sign.items():
                peg_lines[area_name][sign_name][line_num] = {}
                x1 = coords['x'][0]
                x2 = coords['x'][1]
                y1 = coords['y'][0]
                y2 = coords['y'][1]
                a = (y2 - y1) / (x2 - x1) if x1 != x2 else np.inf
                b = y1 - x1 * a
                peg_lines[area_name][sign_name][line_num]['a'] = a
                peg_lines[area_name][sign_name][line_num]['b'] = b

    return peg_x_coords, peg_lines


def get_zone(x, zone='hyper', sample_step=5):
    x *= MMOLL_MGDL
    x = pd.Series(x).astype(int).to_frame()

    if zone.startswith('hyper'):
        if zone == 'hyper L1':
            x = (x >= glucose_levels['hyper L1'][0]) & (x <= glucose_levels['hyper L1'][1])
        elif zone == 'hyper L2':
            x = (x >= glucose_levels['hyper L2'][0]) & (x <= glucose_levels['hyper L2'][1])
        elif zone == 'hyper':
            x = (x >= glucose_levels['hyper L1'][0]) & (x <= glucose_levels['hyper L2'][1])

    elif zone.startswith('hypo'):
        if zone == 'hypo L1':
            x = (x >= glucose_levels['hypo L1'][0]) & (x <= glucose_levels['hypo L1'][1])
        elif zone == 'hypo L2':
            x = (x >= glucose_levels['hypo L2'][0]) & (x <= glucose_levels['hypo L2'][1])
        elif zone == 'hypo':
            x = (x >= glucose_levels['hypo L2'][0]) & (x <= glucose_levels['hypo L1'][1])

    # event time should be at least 15 minutes (so 3 measurements, if sample_step is 5)
    threshold = 15 / sample_step

    mask = True
    for i in range(int(threshold)):
        mask &= x.shift(-i)

    return mask.any().item()


def clarke_error_grid(output, target, units='mmoll'):
    """
    Return masks for points belonging to certain areas in the Clarke Error Grid

    Zone A: Clinically Accurate
        Zone A represents glucose values that deviate from the reference by no more than 20%
        or are in the hypoglycemic range (<70 mg/dl) when the reference is also <70 mg/dl.

    Zone B: Clinically Acceptable
        Upper and lower zone B represents values that deviate from the reference by >20%.

    Zone C: Overcorrecting
        Zone C values would result in overcorrecting acceptable blood glucose levels;
        such treatment might cause the actual blood glucose to fall below 70 mg/dl or rise above 180 mg/dl.

    Zone D: Failure to Detect
        Zone D represents "dangerous failure to detect and treat" errors. Actual glucose values are
        outside of the target range, but patient-generated values are within the target range.

    Zone E: Erroneous treatment
        Zone E is an "erroneous treatment" zone. Patient-generated values within this zone are opposite
        to the reference values, and corresponding treatment decisions would therefore be opposite to that called for.

    Reference:
    [1] Clarke, W.L., Cox, D., Gonder-Frederick, L. A., Carter, W., & Pohl, S. L. (1987).
        "Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose".
        Diabetes Care, 10(5), 622–628. doi: 10.2337/diacare.10.5.622
    [2] Clarke, WL. (2005). "The Original Clarke Error Grid analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
    """
    assert units == 'mmoll' or units == 'mgdl', 'Please give either mmoll or mgdl as units here.'
    if units == 'mmoll':
        target = copy(target) * MMOLL_MGDL
        output = copy(output) * MMOLL_MGDL
    clarke = {'A': (abs(target - output) <= 0.2 * target) | ((target < 70) & (output < 70))}
    clarke['E'] = ((target >= 180) & (output <= 70)) | ((output >= 180) & (target <= 70)) & ~clarke['A']
    clarke['D'] = ((output >= 70) & (output <= 180) & ((target > 240) | (target < 70))) & ~clarke['A'] & ~clarke['E']
    clarke['C'] = ((target >= 70) & (output >= target + 110)) | ((target <= 180) & (output <= (7 / 5) * target - 182)) & ~clarke['A'] & ~clarke['E'] & ~clarke['D']
    clarke['B'] = (abs(target - output) > 0.2 * target) & ~clarke['A'] & ~clarke['E'] & ~clarke['D'] & ~clarke['C']
    return dict(sorted(clarke.items()))


def percent_in_areas(output, target, grid_type='clarke', units='mmoll', pointwise=False):
    if grid_type == 'clarke':
        area_masks = clarke_error_grid(output, target, units=units)
    elif grid_type == 'parkes':
        area_masks = parkes_error_grid(output, target, units=units)
    num_points = list(area_masks.items())[0][1].size()[0]
    tensor_list = [100 * (mask.sum(dim=0) / num_points) for area, mask in area_masks.items()]
    matrix = torch.stack(tensor_list, dim=1)
    if not pointwise:
        matrix = matrix.mean(dim=0)
    matrix = matrix.cpu().detach().numpy()
    return matrix


def parkes_error_grid(output, target, units='mmoll'):
    assert units == 'mmoll' or units == 'mgdl', 'Please give either mmoll or mgdl as units here.'
    if units == 'mmoll':
        target = target * MMOLL_MGDL
        output = output * MMOLL_MGDL

    def above_line(x_1, y_1, x_2, y_2):
        y_line = ((y_1 - y_2) * target + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return output >= y_line

    def below_line(x_1, y_1, x_2, y_2):
        return ~above_line(x_1, y_1, x_2, y_2)

    parkes = {'E': above_line(0, 150, 35, 155) & above_line(35, 155, 50, 550)}
    parkes['D'] = (((output > 100) & above_line(25, 100, 50, 125) & above_line(50, 125, 80, 215) & above_line(80, 215, 125, 550)) |
                   ((target > 250) & below_line(250, 40, 550, 150))) & \
                  ~parkes['E']
    parkes['C'] = (((output > 60) & above_line(30, 60, 50, 80) & above_line(50, 80, 70, 110) & above_line(70, 110, 260, 550)) |
                   ((target > 120) & below_line(120, 30, 260, 130) & below_line(260, 130, 550, 250))) & \
                  ~parkes['E'] & ~parkes['D']
    parkes['B'] = (((output > 50) & above_line(30, 50, 140, 170) & above_line(140, 170, 280, 380)) | ((target >= 280) & above_line(280, 380, 430, 550)) |
                   ((target > 50) & below_line(50, 30, 170, 145) & below_line(170, 145, 385, 300) | (target >= 385) & below_line(385, 300, 550, 450))) & \
                  ~parkes['E'] & ~parkes['D'] & ~parkes['C']
    parkes['A'] = ~parkes['E'] & ~parkes['D'] & ~parkes['C'] & ~parkes['B']
    return dict(sorted(parkes.items()))


def calculate_peg_loss(X, Y, device=torch.device('cpu')):
    """ From 2d numpy tensors X and Y """
    from forecast.loss import PEGLossElementwise
    X_torch, Y_torch = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    X_loss, Y_loss = torch.unsqueeze(X_torch, 2).to(device), torch.unsqueeze(Y_torch, 0).to(device)
    peg_loss_instance = PEGLossElementwise()
    return peg_loss_instance(Y_loss, X_loss).cpu()


def calculate_peg_derivative(X, Y):
    slopes = {'A': 0.1, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    areas = parkes_error_grid(Y, X)

    z = np.zeros_like(X)
    z[areas['A']] = slopes['A']
    z[areas['B']] = slopes['B']
    z[areas['C']] = slopes['C']
    z[areas['D']] = slopes['D']
    z[areas['E']] = slopes['E']
    return z * np.where(Y >= X, 1, -1)
