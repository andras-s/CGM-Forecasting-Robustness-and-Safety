import json
import time
import itertools
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.expanduser("~/Thesis/icarus/"))
from preprocess.impute.main import main
from preprocess.impute.config import args

grid_training_noises = [0.9]
grid_num_mixtures = [m for m in range(20, 32)]

folder_name = f"gridsearch_{time.strftime('%Y%m%d-%H%M%S')}"
args['num_mixtures'] = grid_num_mixtures
args['noise_fixed_train'] = grid_training_noises

if not os.path.exists(os.path.join(args.save_dir, folder_name)):
    os.makedirs(os.path.join(args.save_dir, folder_name))
with open(os.path.join(args.save_dir, folder_name, 'config.txt'), 'w') as file:
    file.write(json.dumps(args))

for num_mixtures, training_noise in tqdm(itertools.product(grid_num_mixtures, grid_training_noises)):
    print(f"\n number of mixtures: {num_mixtures} \n training noise: {training_noise}\n")

    args['num_mixtures'] = num_mixtures
    args['noise_fixed_train'] = training_noise
    args['save_string'] = folder_name + f"/run_mixtures_{args['num_mixtures']}_noise_{args['noise_fixed_train']}"

    try:
        main()
    except:
        continue
