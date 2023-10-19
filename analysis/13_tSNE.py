import random
import numpy as np
from helper import t_SNE_main, create_t_SNE_plots
# TODO: (IDEA for setting the elementary domains manually) visually speaking we would want our domain bases to encircle
#  the FE-outputs. So we could train a FE, (t-SNE or just use the whole of) its output, calculate 'large enough' circle
#  around the (t-SNE or even the whole n-dim) embeddings, reverse calculate what kind of bases would yield those points.
#  !Also! the elementary bases should not just encircle the FE-output, but rather have some overlapping it and some
#  outside (to make it robust)


# Base Settings
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/LSTM_Layer_v2'
save_to = '/local/home/ansass/Thesis/icarus/analysis/results/13_t_SNE/'
gpu = '0'
random.seed(0)
np.random.seed(0)


# # Calculate and plot embedding
# args = {'model_type': 'FT_MMD', 'domain_split': 'HbA1c_level', 'seed': 1, 'horizon': 1, 'fold': 2, 'metric': 'euclidean', 'n_iter': 10000, 'perplexity': 29461, 'lr': 250, 'plot_gdu_bases': True, 'plot_fe_output': True, 'num_fe_outputs': None, 'plot_num_fe_outputs': 'all'}
# t_SNE_main(experiment_path=experiment_path, gpu=gpu, save_to=save_to, **args)


# # Calculate and plot embeddings separately
# args = {'model_type': 'FT_MMD', 'domain_split': 'HbA1c_level', 'seed': 1, 'horizon': 1, 'fold': 2, 'metric': 'euclidean', 'n_iter': 10000, 'perplexity': 300, 'lr': 50, 'plot_gdu_bases': True, 'plot_fe_output': True, 'num_fe_outputs': None, 'plot_num_fe_outputs': 'all'}
# for num_fe_outputs in [None]:  # [500, 1000, 1500, 2000, 5000, None]
#     args["num_fe_outputs"] = num_fe_outputs
#     for perplexity in [10, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000]:
#         args["perplexity"] = perplexity
#         for lr in [10, 100, 250, 500, 1000]:
#             args["lr"] = lr
#
#             t_SNE_main(experiment_path=experiment_path, gpu=gpu, save_to=save_to, **args)

# # Plot embeddings together
# args = {'model_type': 'FT_MMD', 'domain_split': 'HbA1c_level', 'seed': 1, 'horizon': 1, 'fold': 2, 'metric': 'MMD', 'n_iter': 4000, 'perplexities': [25, 50, 100, 150, 250, 500, 1000], 'lrs': [50, 100, 150, 250, 500, 1000], 'plot_gdu_bases': True, 'plot_fe_output': True, 'num_fe_outputs': None, 'plot_num_fe_outputs': 'all'}
args = {'model_type': 'FT_MMD', 'domain_split': 'HbA1c_level', 'seed': 1, 'horizon': 1, 'fold': 2, 'metric': 'euclidean', 'n_iter': 10000, 'perplexities': [10, 100, 250, 500, 1000, 2500, 5000, 10000], 'lrs': [10, 100, 250, 500, 1000], 'plot_gdu_bases': True, 'plot_fe_output': True, 'num_fe_outputs': None, 'plot_num_fe_outputs': 'all'}
for num_fe_outputs in [None]:  # [500, 1000, 1500, 2000, 5000, None]
    args["num_fe_outputs"] = num_fe_outputs
    create_t_SNE_plots(experiment_path=experiment_path, gpu=gpu, save_to=save_to, **args)
