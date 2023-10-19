from helper import get_kpis, calculate_kpi_relative_changes, plot_kpis_for_varying_peg_weight

# INPUTS
experiment_path = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/save/Thesis/PEG_Experiment/04_kpis_with_varying_peg_weight/LSTM'
save_to = '/local/home/ansass/Thesis/icarus/analysis/results/ablation_study'
horizon = 1
peg_weights = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]          # [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64]


# MAIN
# kpi_names = ['test_pointwise NLL', 'test_pointwise RMSE']
# kpi_names = ['test_NLL']
# kpi_names = ['test_RMSE']
# kpi_names = ['test_pointwise PEG [% in A-E]']
# kpi_names = ['test_PEG [% in A-E]']
kpi_names = ['test_NLL', 'test_RMSE', 'test_PEG [% in A-E]']

kpis = get_kpis(experiment_path, horizon, peg_weights, kpi_names)
rel_changes = calculate_kpi_relative_changes(kpis)
plot_kpis_for_varying_peg_weight(rel_changes, seperate_plots=True, save_to=save_to)
