from analysis.helper import load_logs, load_best_model_logs, plot_logs, compare_logs


horizon = 0.5

# specify run location
if horizon == 0.5:
    run_path_nll = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220914-153607_LSTM_i6_hor0.5_b1024_lr0.001_g0.999_seed0_linearlinear'
    run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220914-153720_LSTM_i6_hor0.5_b1024_lr0.001_g0.999_seed0_linearlinear'
elif horizon == 1:
    run_path_nll = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220914-153619_LSTM_i6_hor1_b1024_lr0.001_g0.999_seed0_linearlinear'
    run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220914-153744_LSTM_i6_hor1_b1024_lr0.001_g0.999_seed0_linearlinear'      # weight = 1
    # run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220917-144053_LSTM_NLLPEGSurface_i6_hor1_seed0'                          # weight = 4
    # run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220918-093816_LSTM_NLLPEGSurface_i6_hor1_seed0'                          # weight = 8
    # run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220918-093916_LSTM_NLLPEGSurface_i6_hor1_seed0'                          # weight = 16
    # run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220919-112447_LSTM_NLLPEGSurface_i6_hor1_seed0'                          # weight = 32
elif horizon == 2:
    run_path_nll = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220914-153639_LSTM_i6_hor2_b1024_lr0.001_g0.999_seed0_linearlinear'
    run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220914-153759_LSTM_i6_hor2_b1024_lr0.001_g0.999_seed0_linearlinear'
    # run_path_peg = '/local/home/ansass/Thesis/icarus/forecast/results/rnn/regression/run_20220918-093832_LSTM_NLLPEGSurface_i6_hor2_seed0'                             # weight = 8

# load logs
logs_nll = load_logs(run_path_nll)
logs_peg = load_logs(run_path_peg)
best_model_logs_nll = load_best_model_logs(run_path_nll)
best_model_logs_peg = load_best_model_logs(run_path_peg)

# plot logs
#plot_logs(best_model_logs_nll)
#plot_logs(best_model_logs_peg)

# compare logs
compare_logs(best_model_logs_nll, best_model_logs_peg)

# plot intermediate models forecast
