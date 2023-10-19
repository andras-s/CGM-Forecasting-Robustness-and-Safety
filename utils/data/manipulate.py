import os
import pickle
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import PredefinedSplit
import random

from utils.setup import DATA_DIR


def compute_relative_time(df):
    # compute the relative time between points based on datetime values
    df['relative_time'] = (pd.to_timedelta(df.datetime - df.datetime.iloc[0]).values / 6e10).astype(float)
    return df


def compute_true_time(df):
    df['true_time'] = (pd.to_timedelta(df.datetime.dt.time.astype(str)).values / 6e10).astype(float)
    return df


def split_gaps(df, max_gap='1h'):
    """
    Splits data series into multiple shorter but continuous series if there are gaps that are bigger than max_gap
    """
    df.datetime = pd.to_datetime(df.datetime)
    loc_split = np.where(df.datetime.diff() > max_gap)[0]

    # split df where the periods are bigger than max_gap
    if len(loc_split) == 0:
        return [df]
    else:
        loc_split = np.concatenate(([df.index.min()], loc_split))
        loc_split = np.concatenate((loc_split, [df.index.max()])) if loc_split[-1] != df.index.max() else loc_split
        return [df.iloc[loc_split[i]:loc_split[i + 1]].reset_index(drop=True) for i in range(len(loc_split) - 1)]


def create_windows(data, file_ids, window_length=24, tol_length=1, max_gap='1h', window_start=None):
    """
    Generates (non-overlapping!) windows of specified length from all dataframes in the passed list.

    Parameters
    ----------
    data : list of pd.Dataframes or pd.Dataframe
        list of the dataframes from which the windows should be extracted;
        if a single dataframe is passed it is automatically converted to a list containing itself
    window_length : int, optional
        the length of the windows in hours
    tol_length : float, optional
        determines the minimum length of a time series, i.e. if windows of 24h should be created,
        then windows with length window_length-tolerance will still be accepted. This is important since there might
        be missing values at the beginning or the end of a generated window
    window_start : float, optional
        specifies at what time the windows should start, 
        e.g., if window_start=14, all windows start at 2PM 
        and if None, then windows just start at first measurement
    
    Returns
    -------
    window_list : list
        list of the generated windows
    """

    windows = {}
    print("Cutting data into windows...")
    for (f_id, data_file) in tqdm(zip(file_ids, data)):

        data_file = split_gaps(data_file, max_gap=max_gap)

        windows_file = []

        for df in data_file:
            # check into how many the dataframe needs to be cut
            ts_min = df.datetime.iloc[0]

            if window_start:  # if we want windows to start at a specific time
                if ts_min.hour < window_start:  # if time series starts earlier than when we want it to start, then start at starting time
                    ts_min = pd.to_datetime(ts_min.date()) + pd.to_timedelta(f"{window_start}h")
                else:  # if time series starts later, then start at next day at window_start
                    ts_min = pd.to_datetime(ts_min.date() + pd.to_timedelta("1d")) + pd.to_timedelta(f"{window_start}h")

                freq = '24h'  # frequency at which we sample new windows

            else:  # if we don't want the windows to start at a specific time
                freq = f"{window_length}h"

            # number of patient windows
            n = np.ceil((df.datetime.iloc[-1] - ts_min) / pd.to_timedelta(freq))
            ts_splits = pd.date_range(start=ts_min, periods=n, freq=freq)

            for i in range(len(ts_splits)):
                # splitting at the computed times and reducing the dataframes to the required window length
                w = df.loc[(df.datetime >= ts_splits[i]) & (
                            df.datetime < ts_splits[i] + pd.to_timedelta(f"{window_length}h"))].reset_index(drop=True)

                if not w.empty:
                    w = compute_relative_time(w)
                    w = compute_true_time(w)
                    if w.relative_time.iloc[-1] >= (window_length - tol_length) * 60:
                        # w = fill_time(w)
                        windows_file.append(w)

        windows[f_id] = windows_file
    return windows


def split_by_patient(data: dict, K=5, tolerance=0.01, seed=0):
    """
    Splits data into different folds for cross-validation.
    Splitting is patient-based and takes into account amount of data per patient
    to obtain equal data sizes per fold.

    Parameters
    ----------
    data : dict
        key patient and values list of windows for patient
    K : int
        the number of folds for cross-validation
    tolerance : float
        the windows are not exactly of size 1 / K as the split is done based on patients; the tolerance is
        used to define an upper limit for the size of each fold of (1 + tolerance) / K

    Returns
    -------
    data : list
        list of data
    patients : list of patients
    folds : list
        list of indices which assign each sample to a validation fold
    """
    random.seed(seed)

    patients = list(data.keys())  # list of patients

    N = sum([len(d) for d in data.values()])  # total number of windows

    folds_patients = K * [[]]
    for k in range(K - 1):
        fold = []
        count = 0
        while count < int(N / K):
            pid = random.choice(patients)

            fold.append(pid)
            patients.remove(pid)

            count += len(data[pid])

        # ensure that no fold is too big
        while count / N > (1 + tolerance) / K:
            N_overshoot = count - int(N / K)  # number of windows above supposed size

            # find patient whose number of windows is closest to N_overshoot
            pid = fold[np.abs(np.array([len(data[pid]) for pid in fold], dtype=int) - N_overshoot).argmin()]

            fold.remove(pid)
            patients.append(pid)

            count -= len(data[pid])

        folds_patients[k] = set(fold)

    folds_patients[K - 1] = set(patients)

    return [df for k in range(K) for p in folds_patients[k] for df in data[p]], \
           [p for k in range(K) for p in folds_patients[k] for _ in data[p]], \
           [k for k in range(K) for p in folds_patients[k] for _ in data[p]]


def split_by(data: dict, attribute: str, load_dir='3_clean_data/', K=5, tolerance=0.01, seed=0):
    """
    Splits data into different folds.
    Splitting is attribute based and possible attributes are (treatment, study).

    Parameters
    ----------
    data : dict
        key file and values list of windows generated from file
    attribute : str
        attribute to split by
    K : int
        the number of folds for cross-validation if no attribute is passed
    tolerance : float
        the windows are not exactly of size 1 / K as the split is done based on patients; the tolerance is
        used to define an upper limit for the size of each fold of (1 + tolerance) / K

    Returns
    -------
    data : list
        list of data
    patients : list of patients
    folds : list
        list of indices which assign each sample to a validation fold
    """
    random.seed(seed)
    file_ids = list(data.keys())  # list of patients
    N = sum([len(d) for d in data.values()])  # total number of windows
    folds_file_ids = K * [[]]

    if attribute is None:               # Split by file. Consider each files size.
        for k in range(K - 1):
            fold = []
            count = 0
            while count < int(N / K):
                f_id = random.choice(file_ids)
                fold.append(f_id)
                file_ids.remove(f_id)
                count += len(data[f_id])
            # ensure that no fold is too big
            while count / N > (1 + tolerance) / K:
                N_overshoot = count - int(N / K)  # number of windows above supposed size
                # find patient whose number of windows is closest to N_overshoot
                f_id = fold[np.abs(np.array([len(data[f_id]) for f_id in fold], dtype=int) - N_overshoot).argmin()]
                fold.remove(f_id)
                file_ids.append(f_id)
                count -= len(data[f_id])
            folds_file_ids[k] = set(fold)
        folds_file_ids[K - 1] = set(file_ids)
        split_data = [df for k in range(K) for p in folds_file_ids[k] for df in data[p]]
        file_ids = [p for k in range(K) for p in folds_file_ids[k] for _ in data[p]]
        fold_ids = [k for k in range(K) for p in folds_file_ids[k] for _ in data[p]]

    elif attribute == 'treatment':
        files_info = pd.read_excel(DATA_DIR + load_dir + 'files_info.xlsx')
        print(f'\nSplitting by treatment')
        print(f'treatments found:')
        print(files_info.treatment.value_counts(), '\n')
        split_data = []
        file_ids = []
        fold_ids = []
        for f_id, window_list in data.items():
            num_windows = len(window_list)
            treatment = files_info.loc[files_info.file_id == int(f_id), 'treatment'].item()
            if treatment in ['basal', 'basal only', 'insulin+nia']:
                domain = 0
            elif treatment in ['MDI']:
                domain = 1
            elif treatment in ['CSII', 'Medtronic 640G']:
                domain = 2
            elif treatment in ['HYBRID_mm670g', 'HYBRID_mm780g', 'HYBRID_accucheck']:
                domain = 3
            else:
                domain = 'drop'
                print(f'Dropping file: {f_id} with treatment: {treatment}')
            if domain != 'drop':
                split_data.extend(window_list)
                file_ids.extend(num_windows*[f_id])
                fold_ids.extend(num_windows*[domain])
        fold_ids, split_data, file_ids = (list(z) for z in zip(*sorted(zip(fold_ids, split_data, file_ids), key=lambda tup: tup[0])))
        print(f'\nDomain windows:')
        print(f'0 BI     {fold_ids.count(0)}')
        print(f'1 MDI    {fold_ids.count(1)}')
        print(f'2 CSII   {fold_ids.count(2)}')
        print(f'3 AP     {fold_ids.count(3)}')
    return split_data, file_ids, fold_ids


def generate_predefined_split(data: list, file_ids: list, fold_ids):
    """
    Yields a generator object for cross-validation based on the indices that assign samples to folds.
    This function is needed, since the data is always split using the custom split.
    """
    for idx_train_val, idx_test in PredefinedSplit(fold_ids).split():
        fold_ids_s = pd.Series(fold_ids)
        test_fold_id = fold_ids[idx_test[0]]
        train_val_fold_ids_unique = list(set(fold_ids_s) - set([test_fold_id]))
        train_val_fold_ids_unique.sort()
        num_windows_smallest_fold = fold_ids_s.value_counts().min()
        num_val_windows_per_fold = math.floor((num_windows_smallest_fold * 0.25))
        train_val_indices_per_fold = {f_id: [idx for idx in idx_train_val if fold_ids[idx] == f_id] for f_id in train_val_fold_ids_unique}
        val_indices_per_fold = {f_id: random.sample(train_val_indices_per_fold[f_id], num_val_windows_per_fold) for f_id in train_val_fold_ids_unique}
        train_indices_per_fold = {f_id: list(set(train_val_indices_per_fold[f_id]) - set(val_indices_per_fold[f_id])) for f_id in train_val_fold_ids_unique}

        idx_train = sum(train_indices_per_fold.values(), [])
        idx_train = np.array(idx_train, dtype='int')
        idx_val = sum(val_indices_per_fold.values(), [])
        idx_val = np.array(idx_val, dtype='int')

        yield ([data[i] for i in idx_train], [data[i] for i in idx_val], [data[i] for i in idx_test]), \
              ([file_ids[i] for i in idx_train], [file_ids[i] for i in idx_val], [file_ids[i] for i in idx_test])


def filter_patients(data: list, patients: list, **kwargs):
    print("Filtering patients by ..")
    info = pd.read_excel(DATA_DIR + '/patients.xlsx', engine='openpyxl')

    info.rename(columns={'Code (med=medtronic; dex=dexcom; fs=freestyle libre)': 'code',
                         'device (1=medtronic, 2=dexcom, 3=libre)': 'sensor',
                         'Geschlecht (1=m, 2=f)': 'gender',
                         'Type Diabetes (1=DM1, 2=DM2, 3=GDM, 4=MODY, 5=other)': 'diabetes_type',
                         'Therapy (1=MDI, 2=CSII, 3=no insulin antidiabetics [nia], 4=basal insulin, 5=insulin & nia, 6=other)': 'therapy'},
                inplace=True)
    info = info[['code', 'ID', 'sensor', 'gender', 'diabetes_type', 'therapy']]
    info['sensor'].replace({1: 'medtronic', 2: 'dexcom', 3: 'libre'}, inplace=True)
    info['gender'].replace({1: 'm', 2: 'f'}, inplace=True)
    info['diabetes_type'].replace({1: 'DM1', 2: 'DM2', 3: 'GDM', 4: 'MODY', 5: 'other', 6: 'other', 99: 'other'},
                                  inplace=True)
    info['therapy'].replace(
        {1: 'MDI', 2: 'CSII', 3: 'nia', 4: 'basal insulin', 5: 'insulin+nia', 6: 'other', 99: 'other'}, inplace=True)

    mask = len(info) * [True]
    for k, v in kwargs.items():
        assert v in info[k].unique(), "Argument {v} not in {k} options".format(k=k, v=v)
        print(f".. {k} is {v}")

        mask &= info[k] == v
    patients_subset = info.loc[mask, 'ID'].tolist()

    return [df for df, p in zip(data, patients) if p in patients_subset]
