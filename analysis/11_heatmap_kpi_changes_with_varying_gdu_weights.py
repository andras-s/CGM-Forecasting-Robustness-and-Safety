from helper import get_gdu_kpis, heatmap_kpis_for_varying_gdu_weights

# INPUTS
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/11_loss_weight_ablation'
# experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/Layer_Experiment/11_loss_weight_ablation/archive/recent_with_only_four_weights'
save_to = '/local/home/ansass/Thesis/icarus/analysis/results/ablation_study/gdu_loss_weights_ablation/'
weights = ['0.0', '0.01', '0.1', '1.0', '10.0']
# weights = ['0', '0.01', '0.1', '1']
ols_weights, l1_weights = weights, weights
# kpi_names = ['test_NLL', 'test_RMSE', 'test_PEG [% in A-E]']
kpi_names = ['test_NLL']


# MAIN
kpis = get_gdu_kpis(experiment_path, ols_weights, l1_weights, kpi_names)
heatmap_kpis_for_varying_gdu_weights(kpis, save_to=save_to)
