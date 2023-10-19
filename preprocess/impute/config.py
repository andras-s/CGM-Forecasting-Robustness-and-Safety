# import argparse
import time
from utils.utils import dotdict


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description="RNN model training")

    parser.add_argument('--load_dir', type=str, required=True, help='The directory from were data is loaded')
    parser.add_argument('--sensors', type=str, required=True, help='Only the patients with the specified sensors are loaded')
    parser.add_argument('--save_dir', type=str, required=True, help='The directory where results are saved')
    parser.add_argument('--gpu', type=int, default=None, help='Number of the gpu to use')
    parser.add_argument('--lr', type=float, default=1., help='The initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1500, help='Maximum number of epochs')
    parser.add_argument('--early_stopping', type=int, default=200, help='Number of epochs for early stopping')
    parser.add_argument('--gamma', type=float, default=1.0, help='The rate of decay of the learning rate')
    parser.add_argument('--kernel', type=str, default="RBF", help='Kernel')
    parser.add_argument('--num_mixtures', type=int, default=4, help='Number of subkernels in the spectral mixture kernel')
    parser.add_argument('--mean', type=str, default="linear", help='Mean Module')
    parser.add_argument('--initial_mean', type=float, default=9.4, help='Initialize constant mean')
    parser.add_argument('--noise_fixed_train', type=float, default=0.1, help='Noise of the GP during training is fixed')
    parser.add_argument('--noise_lower_limit_train', type=float, default=0.9, help='Lower limit for the noise of the GP during training')
    parser.add_argument('--noise_upper_limit_train', type=float, default=0.9, help='Upper limit for the noise of the GP during training')
    parser.add_argument('--noise_final_gp', type=float, default=0, help='If not None, then the noise of the trained GPs is set to the specified value')
    parser.add_argument('--initial_frequencies', type=list, default=[], help='Initial frequencies of the GP with SM kernel')
    parser.add_argument('--freeze_frequencies', type=bool, default=True, help='Freeze initial frequencies of the GP with SM kernel')
    parser.add_argument('--frequency_lower_limit', type=float, default=1.0, help='Lower limit for the frequency of the spectral mixture components of the GP')
    parser.add_argument('--frequency_upper_limit', type=float, default=1.0, help='Upper limit for the frequency of the spectral mixture components of the GP')
    parser.add_argument('--initial_decays', type=list, default=[], help='Initial Decays of the GP with SM kernel')
    parser.add_argument('--freeze_decays', type=bool, default=True, help='Freeze initial decays of the GP with SM kernel')
    parser.add_argument('--decay_upper_limit', type=float, default=0.003, help='Upper limit for the decay of the spectral mixture components of the GP')
    parser.add_argument('--grid_size', type=int, default=1000, help='SKI grid-size')
    parser.add_argument('--K', type=float, default=5, help='The number of cross-validation folds')
    parser.add_argument('--window_length', type=int, default=24, help='Length of the time windows (h)')
    parser.add_argument('--window_start', type=int, default=6, help='Start of the time windows (h)')
    parser.add_argument('--force_new_windows', type=bool, default=False, help='Whether to create new windows, even if they already exist')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--calculate_kpis', type=bool, default=False, help='Calculate KPIs of the GP after training')
    parser.add_argument('--tb_logging', action='store_true', help='Whether to log losses in tensorboard')
    parser.add_argument('--plot_style', type=str, default='white', help='Seaborn plotting style')

    return parser.parse_args()


# parse arguments
# args = parse_args()

args = dict(load_dir='//',
            sensors='nonAP',  # options: 'all', 'nonAP', 'AP', 'preAP', 'postAP', sublist from ["med", "dex", "DIA", "MM670G", "MM640G", "FSL", "DEXCOM", "WCA", "AP", "preAP", "postAP"]
            save_dir='andras/results/',
            gpu='1',
            lr=1,
            batch_size=512,
            epochs=1,
            early_stopping=200,
            gamma=0.9,  # 0.99
            kernel='SM',
            num_mixtures=5,  # only relevant for spectral mixture (SM) kernel
            mean="constant",
            initial_mean=9.4,

            noise_fixed_train=0.9,
            noise_lower_limit_train=0.05,  # 0.1, 0.9
            noise_upper_limit_train=0.1,  # 0.9
            noise_final_gp=0.,  # 0.9 (if 0, then noise is not reset after training)

            # initial_frequencies=[1 / (freq * 60) for freq in [4, 8, 12, 24]],  # [1/(freq*60) for freq in [24, 12, 6, 4, 3, 2, 1, 0.5]]
            initial_frequencies=[],  # [1/(freq*60) for freq in [24, 12, 6, 4, 3, 2, 1, 0.5]]
            freeze_frequencies=False,
            frequency_lower_limit=1/(60 * 24),
            frequency_upper_limit=1/(60 * 0.5),

            initial_decays=[],  # [0.0023, 0.0024, 0.0025, 0.0027, 0.0028, 0.0031, 0.0050]
            freeze_decays=False,
            decay_upper_limit=0.0162,

            grid_size=1000,
            split_attribute='treatment',  # None for CV. 'treatment',
            K=5,
            window_length=24,  # 24
            max_gap='1h',  # '1h'
            tol_length = 1,  # 1
            window_start=None,  # 6
            seed=0,
            calculate_kpis=True,
            tb_logging=True,
            plot_style='darkgrid')

args['save_string'] = f"run_{args['kernel']}_{time.strftime('%Y%m%d-%H%M%S')}"
if args['split_attribute'] == 'treatment':
    args['K'] = 4
args = dotdict(args)
