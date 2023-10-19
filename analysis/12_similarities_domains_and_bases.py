from helper import get_run_path, get_vectors, categorize_vectors, get_similarities, create_similarity_heatmap
import torch

# Model Settings
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/LSTM_Layer_v2'
model_type = 'FT_MMD'
horizon = 1
fold = 2
seed = 1

domain_split = 'glucose_level_last_input'   # treatment, HbA1c_level, glucose_level_last_input
gpu = '0'
save_to = f'/local/home/ansass/Thesis/icarus/analysis/results/12_similarity_heatmap/similarity_heatmap_seed{seed}.png'

# Main
run_path = get_run_path(experiment_path, model_type, seed, horizon)
raw_vectors = get_vectors(run_path, model_type, horizon, fold, seed, gpu, gdu_bases=True, fe_output=True)
vectors = categorize_vectors(raw_vectors, domain_split)
similarities_df, save_path = get_similarities(vectors, model_type, horizon, fold, seed, domain_split)
create_similarity_heatmap(-similarities_df, save_path)
