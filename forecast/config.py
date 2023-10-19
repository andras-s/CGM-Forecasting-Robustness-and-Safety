import argparse
import time
from utils.utils import dotdict


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument('--repo_location', type=str, required=True, help='The directory containing the project')
    parser.add_argument('--load_dir', type=str, required=True, help='The directory containing the cleaned patient data')
    parser.add_argument('--features', type=list, default=[], help='The features to use besides glucose value')
    parser.add_argument('--mtype', type=str, required=True, help='The type of analysis to do (either classification or regression).')
    parser.add_argument('--model', type=str, required=True, help='The type of model to use.')
    parser.add_argument('--base_dir', type=str, required=True, help='The base directory where results are saved')
    parser.add_argument('--input_duration', type=float, default=6, help='The input duration (h)')
    parser.add_argument('--target_duration', type=float, default=2, help='The prediction horizon (h)')
    parser.add_argument('--pred_start', type=float, default=22, help='The time (h) from which we want to make the prediction')
    parser.add_argument('--sample_step', type=float, default=5, help='The interval (min) with which we resample from the GP')
    parser.add_argument('--loss', type=str, default="NLL", help='The loss criterion used for training.')
    parser.add_argument('--lr_scheduler', type=str, default="ReduceLROnPlateau", help='The learning rate scheduler to use.')
    parser.add_argument('--threshold', type=int, default=.5, help='The decision threshold.')
    parser.add_argument('--LSTM_hps', type=dict, default={}, help='The parameters of the LSTM model.')
    parser.add_argument('--TFT_hps', type=dict, default={}, help='The parameters of the TFT model.')
    parser.add_argument('--Layer_hps', type=dict, default={}, help='The parameters of the Layer model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=list, default=None, help='Numbers of the gpus to use.')
    parser.add_argument('--gpu_ids', type=int, default=None, help='Number of the gpu to use.')
    parser.add_argument('--tb_logging', action='store_true', help='Whether to log losses in tensorboard.')
    parser.add_argument('--domain', type=str, default="sensor", help='Which domain to test on.')
    parser.add_argument('--gp_run_name', type=str, default="run_SM_20220413-163229", help='Which GPs to use.')
    parser.add_argument('--resampling_method', type=list, default=['linear', 'linear'], help='Which resampling methods to use.')
    parser.add_argument('--cut_input_target_at_measurement', type=bool, default=True, help='Whether to cut the input and target at a measurement or not.')
    parser.add_argument('--force_new_imputation', type=bool, default=False, help='Whether to impute data, even if it already exists.')

    return parser.parse_args()


args = dict(repo_location       = "C:/Users/sass9/PycharmProjects/CGM-Forecasting-Robustness-and-Safety/",       # append repo location to $PYTHONPATH such that imports work
            load_dir            = '3_clean_data/',          # 3_poc_data/   # 3_clean_data/
            base_dir            = './results/',             # The base directory where results are saved

            input_duration      = 6,                        # The input duration (h) of the RNN             6
            target_duration     = 1,                        # The prediction horizon (h) of the RNN         2
            pred_start          = None,                     # The time (h) from which we want to make the prediction
            sample_step         = 5,                        # The interval (min) with which we resample from the GP
            gp_run_name         = 'optimal_gp',             # Name of the GP to use for interpolation
            resampling_method   = ['linear', 'linear'],     # Methods to interpolate the inputs and targets
            cut_input_target_at_measurement = True,         # Whether to cut the input and target at a measurement or not
            force_new_imputation = False,                   # Whether to impute data, even if it already exists
            features             = [                        # The features to use besides glucose value ('time', 'glucose_delta', ...)#

            ],

            mtype           = 'regression',                 # The type of analysis to do (classification or regression)
            model           = 'Layer',                       # The type of model to use (LSTM/ConvT/TFT/Layer/Ensemble/t_0/Baseline)
            loss            = 'LayerLoss',                  # The loss criterion used for training (NLL, NLLPEGSurface, LayerLoss)
            PEG_weight      = 1,                            # if NLLPEGSurface is chosen, the weight of the PEG-Loss term is specified here
            lr_scheduler    = 'ExponentialLR',              # The learning rate scheduler to use ('ReduceLROnPlateau', 'ExponentialLR')
            LSTM_hps        = {                             # The parameters of the LSTM model
                'hidden': 256,
                'batch_size': 1024,
                'num_layers': 1,
                'lin_layers': (512, 256),
                'dropouts': (0, 0),
                'lr': 1e-3,                                 # 1e-3
                'gamma': 0.999,                             # 0.999
                'factor': 0.5,
                'patience': 10,
                'epochs': 10000,                            # 1500, 3000, 10000
                'early_stopping': 200
            },
            ConvT_hps       = {                             # The parameters of the Convolutional Transformer model
                'sub_len': None,
                'q_len': 4,
                'n_embd': 8,                               # in paper: 20
                'n_head': 1,                                # in paper: 8
                'num_layer': 1,                             # in paper: 3
                'scale_att': True,
                'dropout': 0,                             # in paper but in code unused: 0.1
                'lin_layers': (512, 256),
                'lin_dropouts': (0, 0),
                'additional_params': {},

                'batch_size': 1024,                         # 128
                'lr': 0.001,                                # 0.001
                'gamma': 0.999,
                'factor': 0.5,
                'patience': 10,
                'epochs': 10000,                             # 500
                'early_stopping': 200
            },
            TFT_hps         = {                             # The parameters of the Transformer model
                'hidden_size': 2,
                'attention_head_size': 1,
                'dropout': 0,
                'hidden_continuous_size': 128,
                'batch_size': 1024,                         # 128
                'lr': 0.001,                                # 0.001
                'gamma': 0.999,
                'factor': 0.5,
                'patience': 10,
                'epochs': 10000,                             # 500
                'early_stopping': 200
            },
            Layer_hps       = {                             # The parameters of the Layer model
                'feature_extractor': 'LSTM',
                'fine_tuning_FEs_path': None,                # 'LSTM_single/Feature_Extractors_LSTM_0_treatment_20220817-163958/LSTM_0_2_treatment_20220817-163958',
                'num_gdus': 9,
                'domain_dim': 10,
                'sigma': 9.5,
                'similarity_measure_name': 'MMD',     # MMD, CS, Projected
                'lambda_OLS': 1,
                'lambda_orth': 1,
                'lambda_sparse': 1,
                'softness_param': 1,
                'batch_size': 1024,                   # 1024
                'lr': 0.001,
                'gamma': 0.999,
                'factor': 0.5,
                'patience': 10,
                'epochs': 200,
                'early_stopping': 200
            },
            Ensemble_hps       = {                             # The parameters of the Ensemble models
                'feature_extractor': 'LSTM',
                'fine_tuning_FEs_path': None,   # None, 'LSTM_single/Feature_Extractors_LSTM_0_treatment_20220817-163958/LSTM_0_2_treatment_20220817-163958'
                'num_heads': 9,
                'batch_size': 1024,
                'lr': 0.001,
                'gamma': 0.999,
                'factor': 0.5,
                'patience': 10,
                'epochs': 10000,
                'early_stopping': 200
            },
            threshold       = .5,                       # The decision threshold
            gpu             = '0',                      # Index of the GPU to use
            seed            = 0,                        # Random seed for reproducibility
            tb_logging      = False,                    # Whether to log losses in tensorboard
            domain          = 'treatment',              # Evaluation across domains
            extensive_logging = False,
            save_models     = True,    # True
            create_plots    = False,    # True
            plot_show       = False
            )

args['run_start']                   = time.strftime('%Y%m%d-%H%M%S')
args['lazy_learner']                = args['model'] not in ['LSTM', 'ConvT', 'TFT', 'Layer', 'Ensemble']
args['batch_size']                  = 2048 if args['lazy_learner'] else args[args['model'] + '_hps']['batch_size']
args['lr']                          = 1 if args['lazy_learner'] else args[args['model'] + '_hps']['lr']
args['gamma']                       = 0.999 if args['lazy_learner'] else args[args['model'] + '_hps']['gamma']
args['factor']                      = 0.1 if args['lazy_learner'] else args[args['model'] + '_hps']['factor']
args['patience']                    = 10 if args['lazy_learner'] else args[args['model'] + '_hps']['patience']
args['epochs']                      = 1 if args['lazy_learner'] else args[args['model'] + '_hps']['epochs']
args['early_stopping']              = 2 if args['lazy_learner'] else args[args['model'] + '_hps']['early_stopping']
hp_string                           = "" if args['lazy_learner'] else f"b{int(args['batch_size'])}_lr{args['lr']}_g{args['gamma']}_"
peg_weight                          = str(args['PEG_weight']) if args['loss'] == 'NLLPEGSurface' else ''
args['save_string']                 = f"run_{args['run_start']}" + '_' + args['model'] + f"_{args['loss']}" + peg_weight + f"_i{args['input_duration']}_hor{args['target_duration']}_seed{int(args['seed'])}"
args['pytorch_forecasting']         = True if args['model'] == 'TFT' else False
args['Layer_hps']['orthogonal_loss']  = True if args['Layer_hps']['similarity_measure_name'] == 'Projected' and args['Layer_hps']['lambda_orth'] > 0 else False
args['Layer_hps']['sparse_coding']  = True if args['Layer_hps']['similarity_measure_name'] == 'Projected' and args['Layer_hps']['lambda_sparse'] > 0 else False

args = dotdict(args)
assert args.mtype in ('classification', 'regression')
