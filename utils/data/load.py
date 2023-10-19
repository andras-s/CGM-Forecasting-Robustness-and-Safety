import os
import pandas as pd
from tqdm import tqdm
from utils.setup import DATA_DIR


def read_data(load_dir="3_clean_data/"):
    """
    Loads cleaned patient data
    """
    path = os.path.join(DATA_DIR, load_dir)

    filenames = [f for f in os.listdir(path) if not (f.startswith('.') or f.startswith('~') or f.startswith('files_info'))]
    file_ids = [f.split('.')[0] for f in filenames]

    print("Loading files")
    df_list = [pd.read_csv(path+f) for f in tqdm(filenames)]

    return df_list, file_ids
