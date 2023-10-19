import os
import sys
import datetime
import random
import shutil

import pandas as pd
import numpy as np

from natsort import natsorted
from tabulate import tabulate

from utils.calc import SENSOR_INTERVAL, MMOLL_MGDL
from utils.setup import DATA_DIR

sys.path.append(os.path.expanduser("~/icarus/"))  # append repo location to $PYTHONPATH such that imports work


def create_patients_df():
    """
        Creates a DataFrame, representing all patients (AP and non-AP). Each row in the DataFrame corresponds to one
        patient (ID, sensor, filenames, code). All patient IDs from load_dir are searched for in patients.xlsx and the
        corresponding ID, sensor, filenames and code are read from patients.xlsx. Similarly, the patients from
        patients_ap.xlsx are read and written to the final df.
        :return: DataFrame representing all patients
    """
    # non-AP patients
    non_ap_filenames = [f.split('.')[0] for f in os.listdir(DATA_DIR + "raw_data/") if not f.startswith(('.', '~'))]
    non_ap_patient_files = pd.read_excel(DATA_DIR + '/patients.xlsx', engine='openpyxl')
    non_ap_patient_files = non_ap_patient_files.rename(columns={'Code (med=medtronic; dex=dexcom; fs=freestyle libre)': 'code',
                                                                'device (1=medtronic, 2=dexcom, 3=libre)': 'sensor'})
    non_ap_patient_files['sensor'] = non_ap_patient_files['sensor'].replace({1: 'medtronic', 2: 'dexcom', 3: 'libre'})
    non_ap_patient_ids = non_ap_patient_files.loc[non_ap_patient_files['code'].isin(non_ap_filenames), "ID"].unique()
    non_ap_patient_sensors = [non_ap_patient_files.loc[non_ap_patient_files['ID'] == patient_id, 'sensor'].unique().tolist() for patient_id in non_ap_patient_ids]
    assert sum([len(sensors) for sensors in non_ap_patient_sensors]) == len(non_ap_patient_sensors)
    non_ap_patients = pd.DataFrame({'ID': non_ap_patient_ids,
                                    'sensor': [non_ap_patient_files.loc[non_ap_patient_files['ID'] == patient_id, 'sensor'].iloc[0] for patient_id in non_ap_patient_ids],
                                    'files': [[file for file in non_ap_filenames if int(file.split("_")[2]) == patient_id] for patient_id in non_ap_patient_ids] })
    non_ap_patients["code"] = ["_".join(p_f[0].split("_")[:-1]) for p_f in non_ap_patients["files"]]

    # AP patients
    ap_patients = pd.read_excel(DATA_DIR + '/patients_ap.xlsx', engine='openpyxl', index_col=0)
    ap_patients["old_ID"] = ap_patients["ID"].copy()
    ap_patients["ID"] = ap_patients["old_ID"].str.replace("ID", "").astype(int) + 442   # 442 = number of patients from other datasource
    ap_patients["files"] = [filename.replace("_RawData", "") for filename in ap_patients["file"]]
    ap_patients["ap_status"] = ap_patients["code"].copy()
    ap_patients.loc[ap_patients["ap_status"] == "after", "ap_status"] = "post"
    ap_patients["code"] = ap_patients["ap_status"] + "AP_" + ap_patients["sensor"] + "_" + ap_patients["ID"].astype("string")

    patients = pd.concat([non_ap_patients, ap_patients]).sort_values(by='ID').reset_index(drop=True)

    patients["num_measurements_pre"] = 0  # for all patients
    patients["num_measurements_post"] = 0  # for all patients
    patients["percent_measurements_post"] = 0  # for all patients

    patients["num_duplicates_pre"] = 0  # for all patients
    patients["percent_duplicates_pre"] = 0  # for all patients
    patients["num_duplicates_post"] = 0  # for all patients
    patients["percent_duplicates_post"] = 0  # for all patients
    patients["num_duplicates_dropped"] = 0  # for all patients
    patients["percent_duplicates_dropped"] = 0  # for all patients
    # patients["num_duplicate_measurements_dropped"] = 0  # for all patients
    # patients["percent_duplicate_measurements_dropped"] = 0  # for all patients

    patients["num_backwards_jumps_pre"] = 0  # for all patients
    patients["num_backwards_jumps_post"] = 0  # for all patients

    patients["num_jump_entries_pre"] = 0  # for all patients
    patients["num_jump_entries_post"] = 0  # for all patients
    patients["num_backwards_jump_entries_dropped"] = 0  # for all patients
    patients["percent_backwards_jump_entries_dropped"] = 0  # for all patients

    patients["num_high_density_timestamps_pre"] = 0
    patients["percent_high_density_timestamps_pre"] = 0
    patients["num_high_density_timestamps_post"] = 0
    patients["percent_high_density_timestamps_post"] = 0
    patients["num_high_density_timestamps_dropped"] = 0
    patients["percent_high_density_timestamps_dropped"] = 0

    patients["num_duplicate_timestamps_pre"] = 0
    patients["percent_duplicate_timestamps_pre"] = 0
    patients["num_duplicate_timestamps_post"] = 0
    patients["percent_duplicate_timestamps_post"] = 0
    patients["num_duplicate_timestamps_dropped"] = 0
    patients["percent_duplicate_timestamps_dropped"] = 0

    patients["num_pisa_pre"] = 0  # for all patients
    patients["percent_pisa_pre"] = 0  # for all patients
    patients["num_pisa_post"] = 0  # for all patients
    patients["percent_pisa_post"] = 0  # for all patients
    patients["num_pisa_dropped"] = 0  # for all patients
    patients["percent_pisa_dropped"] = 0  # for all patients

    patients["num_hypoglycemia_measurements_pre"] = 0  # for all patients
    patients["num_hypoglycemia_measurements_post"] = 0  # for all patients
    patients["num_hyperglycemia_measurements_pre"] = 0  # for all patients
    patients["num_hyperglycemia_measurements_post"] = 0  # for all patients

    patients["num_scans_pre"] = 0   # for libre and FSL patients
    patients["percent_scans_pre"] = 0   # for libre and FSL patients
    patients["num_scans_post"] = 0  # for libre and FSL patients
    patients["percent_scans_post"] = 0  # for libre and FSL patients
    patients["num_scans_dropped"] = 0  # for libre and FSL patients
    patients["percent_scans_dropped"] = 0  # for libre and FSL patients

    return patients


def create_files_df(save_to=None):
    """
        Creates a DataFrame, representing all files. Each row in the DataFrame corresponds to one cleaned file
        (file_id, study, patient_id_study, filename, raw_filename, wca_raw_filenames, raw_file_format, sensor, treatment, sex, age,
        diabetes_type, diabetes_duration, hba1c). All files from load_dir are searched for in the corresponding
        patients_cleaned.csv files.

        :return: DataFrame representing all files
    """

    # id_mapping = pd.read_csv(DATA_DIR + '/0_raw_data/ID.csv', sep=',', header=0)

    # WCA files
    wca_raw_filenames = [f for f in os.listdir(DATA_DIR + "/0_raw_data/WCA/raw/") if not f.startswith(('.', '~'))]
    wca_patient_study_ids = natsorted(list(set(["_".join(filename.split("_")[:-1]) for filename in wca_raw_filenames])))
    wca_clean_file_ids = range(len(wca_patient_study_ids))
    wca_files_info = pd.DataFrame(data=wca_clean_file_ids, columns=['file_id'])
    wca_files_info['study'] = 'wca'
    wca_files_info['patient_id_study'] = pd.Series(wca_patient_study_ids)
    wca_files_info['filename'] = wca_files_info['file_id'].astype(str) + '.csv'
    wca_files_info['raw_filename'] = wca_files_info['patient_id_study'] + '.csv'
    wca_files_info['wca_raw_filenames'] = [[filename for filename in wca_raw_filenames if patient_id+'_' in filename] for patient_id in wca_files_info['patient_id_study']]
    wca_patients_cleaned = pd.read_csv(DATA_DIR + '/0_raw_data/WCA/patients_clean.csv', sep=',', header=0)
    wca_patients_cleaned = wca_patients_cleaned.drop_duplicates(subset='STUDY_ID', keep="last").reset_index(drop=True)
    wca_files_info = pd.merge(wca_files_info, wca_patients_cleaned, left_on='patient_id_study', right_on='STUDY_ID', how='left')
    wca_files_info['raw_file_format'] = wca_files_info['sensor']
    wca_files_info.rename(columns={'age_at_visit [y]': 'age',
                                   'diabetes_duration [y]': 'diabetes_duration',
                                   'HbA1c [%]': 'hba1c'}, inplace=True)
    wca_files_info = wca_files_info[['file_id', 'study', 'patient_id_study', 'filename', 'raw_filename', 'wca_raw_filenames', 'raw_file_format', 'sensor', 'treatment', 'diabetes_type', 'diabetes_duration', 'hba1c', 'sex', 'age']]

    # HYBRID1 files
    h1_raw_filenames = natsorted([f for f in os.listdir(DATA_DIR + "/0_raw_data/HYBRID1/proc/") if not f.startswith(('.', '~'))])
    h1_patient_study_ids = ['SWITCH_' + filename.split("ID")[-1].split('.')[0] for filename in h1_raw_filenames]
    h1_clean_file_ids = [wca_clean_file_ids[-1] + 1 + h1_num for h1_num in range(len(h1_raw_filenames))]
    h1_files_info = pd.DataFrame(data=h1_clean_file_ids, columns=['file_id'])
    h1_files_info['study'] = 'hybrid1'
    h1_files_info['filename'] = h1_files_info['file_id'].astype(str) + '.csv'
    h1_files_info['raw_filename'] = h1_raw_filenames
    h1_files_info['patient_id_study'] = h1_patient_study_ids
    h1_patients_cleaned = pd.read_csv(DATA_DIR + '/0_raw_data/HYBRID1/patients_clean.csv', sep=',', header=0)
    h1_files_info = pd.merge(h1_files_info, h1_patients_cleaned, left_on='raw_filename', right_on='file', how='left')
    h1_files_info['raw_file_format'] = h1_files_info['sensor']
    h1_files_info.loc[h1_files_info.AP == 'POST', 'raw_file_format'] = 'MM670G'
    h1_files_info.loc[h1_files_info.AP == 'PRE', 'raw_file_format'] = 'MM640G'
    h1_files_info.loc[h1_files_info.raw_filename.str.contains('pre_Dexcom'), 'raw_file_format'] = 'DEXCOM'
    h1_files_info.loc[h1_files_info.raw_filename.str.contains('FSL'), 'raw_file_format'] = 'FSL'
    h1_files_info.rename(columns={'age_at_visit [y]': 'age',
                                  'diabetes_duration [y]': 'diabetes_duration',
                                  'HbA1c': 'hba1c'}, inplace=True)
    h1_files_info = h1_files_info[['file_id', 'study', 'patient_id_study', 'filename', 'raw_filename', 'raw_file_format', 'sensor', 'treatment', 'diabetes_type', 'diabetes_duration', 'hba1c', 'sex', 'age']]

    # HYBRID2 files
    h2_raw_filenames = natsorted([f for f in os.listdir(DATA_DIR + "/0_raw_data/HYBRID2/proc/") if not f.startswith(('.', '~'))])
    h2_patient_study_ids = ['SWITCH_' + filename.split("ID")[-1].split('.')[0] for filename in h2_raw_filenames]
    h2_clean_file_ids = [h1_clean_file_ids[-1] + 1 + h2_num for h2_num in range(len(h2_raw_filenames))]
    h2_files_info = pd.DataFrame(data=h2_clean_file_ids, columns=['file_id'])
    h2_files_info['study'] = 'hybrid2'
    h2_files_info['filename'] = h2_files_info['file_id'].astype(str) + '.csv'
    h2_files_info['raw_filename'] = h2_raw_filenames
    h2_files_info['patient_id_study'] = h2_patient_study_ids
    h2_patients_cleaned = pd.read_csv(DATA_DIR + '/0_raw_data/HYBRID2/patients_clean.csv', sep=',', header=0)
    h2_files_info = pd.merge(h2_files_info, h2_patients_cleaned, left_on='raw_filename', right_on='file', how='left')
    h2_files_info['raw_file_format'] = h2_files_info['sensor']
    h2_files_info.loc[h2_files_info.AP == 'POST', 'raw_file_format'] = 'MM780G'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_mm670g'), 'raw_file_format'] = 'MM670G'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_mm640g'), 'raw_file_format'] = 'MM640G'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_Dexcom_mmparadigm'), 'raw_file_format'] = 'libre_dexcom_mmparadigm'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_Dexcom_MDI_ID111'), 'raw_file_format'] = 'dexcom_MDI_111'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_Dexcom_MDI_ID115'), 'raw_file_format'] = 'libre_dexcom'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_FSL_mm640g'), 'raw_file_format'] = 'libre_mm640g'
    h2_files_info.loc[h2_files_info.raw_filename.str.contains('pre_FSL_accucheck'), 'raw_file_format'] = 'libre_accucheck'
    h2_files_info.rename(columns={'age_at_visit [y]': 'age',
                                  'diabetes_duration [y]': 'diabetes_duration',
                                  'HbA1c': 'hba1c'}, inplace=True)
    h2_files_info = h2_files_info[['file_id', 'study', 'patient_id_study', 'filename', 'raw_filename', 'raw_file_format', 'sensor', 'treatment', 'diabetes_type', 'diabetes_duration', 'hba1c', 'sex', 'age']]

    # DIA-HACK files
    dh_patients_cleaned = pd.read_csv(DATA_DIR + '/0_raw_data/DIA-HACK/patients_clean.csv', sep=',', header=0)
    dh_raw_filenames = set(dh_patients_cleaned.file)
    dh_all_raw_filenames = set([f for f in os.listdir(DATA_DIR + "/0_raw_data/DIA-HACK/raw/") if not f.startswith(('.', '~'))])
    dh_raw_filenames = natsorted(list(dh_raw_filenames.intersection(dh_all_raw_filenames)))

    dh_patient_study_ids = ["_".join(filename.split("_")[:-2]) for filename in dh_raw_filenames]
    dh_clean_file_ids = [h2_clean_file_ids[-1] + 1 + dh_num for dh_num in range(len(dh_raw_filenames))]
    dh_files_info = pd.DataFrame(data=dh_clean_file_ids, columns=['file_id'])
    dh_files_info['study'] = 'dia_hack'
    dh_files_info['filename'] = dh_files_info['file_id'].astype(str) + '.csv'
    dh_files_info['raw_filename'] = dh_raw_filenames
    dh_files_info['patient_id_study'] = dh_patient_study_ids
    dh_files_info = pd.merge(dh_files_info, dh_patients_cleaned, left_on='raw_filename', right_on='file', how='left')
    dh_patients = pd.read_excel(DATA_DIR + '/0_raw_data/DIA-HACK/patients.xlsx', header=0)
    dh_patients = dh_patients[['study_id', 'A1c_percent']]
    dh_files_info = pd.merge(dh_files_info, dh_patients, left_on='patient_id_study', right_on='study_id', how='left')
    dh_files_info['raw_file_format'] = dh_files_info['sensor']
    dh_files_info.rename(columns={'age_at_visit [y]': 'age',
                                  'diabetes_duration [y]': 'diabetes_duration',
                                  'A1c_percent': 'hba1c'}, inplace=True)
    dh_files_info = dh_files_info[['file_id', 'study', 'patient_id_study', 'filename', 'raw_filename', 'raw_file_format', 'sensor', 'treatment', 'diabetes_type', 'diabetes_duration', 'hba1c', 'sex', 'age']]

    files_info = pd.concat([wca_files_info, h1_files_info, h2_files_info, dh_files_info]).sort_values(by='file_id').reset_index(drop=True)
    files_info['diabetes_type'] = files_info.diabetes_type.str.replace(pat='1.0', repl='1')

    if save_to:
        files_info.to_excel(save_to+'files_info.xlsx', index=False)

    files_info["cleaned"] = False  # for all files_info

    files_info["num_measurements_pre"] = 0  # for all files_info
    files_info["num_measurements_post"] = 0  # for all files_info
    files_info["percent_measurements_post"] = 0  # for all files_info

    files_info["num_duplicates_pre"] = 0  # for all files_info
    files_info["percent_duplicates_pre"] = 0  # for all files_info
    files_info["num_duplicates_post"] = 0  # for all files_info
    files_info["percent_duplicates_post"] = 0  # for all files_info
    files_info["num_duplicates_dropped"] = 0  # for all files_info
    files_info["percent_duplicates_dropped"] = 0  # for all files_info
    # files_info["num_duplicate_measurements_dropped"] = 0  # for all files_info
    # files_info["percent_duplicate_measurements_dropped"] = 0  # for all files_info

    files_info["num_backwards_jumps_pre"] = 0  # for all files_info
    files_info["num_backwards_jumps_post"] = 0  # for all files_info

    files_info["num_jump_entries_pre"] = 0  # for all files_info
    files_info["num_jump_entries_post"] = 0  # for all files_info
    files_info["num_backwards_jump_entries_dropped"] = 0  # for all files_info
    files_info["percent_backwards_jump_entries_dropped"] = 0  # for all files_info

    files_info["num_high_density_timestamps_pre"] = 0
    files_info["percent_high_density_timestamps_pre"] = 0
    files_info["num_high_density_timestamps_post"] = 0
    files_info["percent_high_density_timestamps_post"] = 0
    files_info["num_high_density_timestamps_dropped"] = 0
    files_info["percent_high_density_timestamps_dropped"] = 0

    files_info["num_duplicate_timestamps_pre"] = 0
    files_info["percent_duplicate_timestamps_pre"] = 0
    files_info["num_duplicate_timestamps_post"] = 0
    files_info["percent_duplicate_timestamps_post"] = 0
    files_info["num_duplicate_timestamps_dropped"] = 0
    files_info["percent_duplicate_timestamps_dropped"] = 0

    files_info["num_pisa_pre"] = 0  # for all files_info
    files_info["percent_pisa_pre"] = 0  # for all files_info
    files_info["num_pisa_post"] = 0  # for all files_info
    files_info["percent_pisa_post"] = 0  # for all files_info
    files_info["num_pisa_dropped"] = 0  # for all files_info
    files_info["percent_pisa_dropped"] = 0  # for all files_info

    files_info["num_hypoglycemia_measurements_pre"] = 0  # for all files_info
    files_info["num_hypoglycemia_measurements_post"] = 0  # for all files_info
    files_info["num_hyperglycemia_measurements_pre"] = 0  # for all files_info
    files_info["num_hyperglycemia_measurements_post"] = 0  # for all files_info

    files_info["num_scans_pre"] = 0   # for libre and FSL files_info
    files_info["percent_scans_pre"] = 0   # for libre and FSL files_info
    files_info["num_scans_post"] = 0  # for libre and FSL files_info
    files_info["percent_scans_post"] = 0  # for libre and FSL files_info
    files_info["num_scans_dropped"] = 0  # for libre and FSL files_info
    files_info["percent_scans_dropped"] = 0  # for libre and FSL files_info

    return files_info


def get_jump_indices(df):
    """
    Identifies number and position of entries where the datetime does not follow the expected order.
    """
    df["dt_diffs"] = df["datetime"].diff(1)
    jump_indices = df[df["dt_diffs"] < "0min"].index.to_list()
    num_jumps = len(jump_indices)

    overlap_end_indices = [index - 1 for index in jump_indices]
    overlap_start_datetimes = df.loc[jump_indices, "datetime"].to_list()
    overlap_end_datetimes = df.loc[overlap_end_indices, "datetime"]
    maximum_overlap_start_datetimes = (overlap_end_datetimes - datetime.timedelta(hours=1)).to_list()
    remove_start_datetimes = pd.DataFrame([overlap_start_datetimes, maximum_overlap_start_datetimes]).min()
    remove_end_datetimes = overlap_end_datetimes

    df["overlap"] = False
    overlap_positions = df["overlap"]
    df["overlap"] = False
    for overlap in range(len(remove_start_datetimes)):
        overlap_positions = overlap_positions | ((df.datetime >= remove_start_datetimes.iloc[overlap]) & (
                df.datetime <= remove_end_datetimes.iloc[overlap]))

    num_overlap_entries = overlap_positions.sum()

    return jump_indices, num_jumps, overlap_positions, num_overlap_entries


def get_high_density_indices(df, sensor):
    """
    Identifies number and position of entries where more measurements are in a region than expected.
    """
    ### Identify regions where 3 measurements are within interval+2 minutes
    df = df.sort_values(by='datetime', ignore_index=True)
    interval = SENSOR_INTERVAL[sensor.upper()]
    shifted_datetimes = df.datetime.shift(-2)
    timedeltas_shifted_datetimes = shifted_datetimes - df.datetime
    high_density_locations = timedeltas_shifted_datetimes <= f"{interval + 2} min"
    high_density_locations = high_density_locations | \
                             high_density_locations.shift(1, fill_value=False) | \
                             high_density_locations.shift(2, fill_value=False)
    num_high_density_locations = high_density_locations.sum()
    high_density_indices = df[high_density_locations].index.to_list()

    return high_density_indices, num_high_density_locations


def get_pisa_indices(df, method="cautious"):
    """
    Identifies number and position of entries which are probably affected by PISA.
    """
    if method == "cautious":
        roc_in_thresh = -3.1 / MMOLL_MGDL
        roc_out_thresh = -2.8 / MMOLL_MGDL
        roc_ratio_thresh = 1.3
    elif method == "nominal":
        roc_in_thresh = -2.5 / MMOLL_MGDL
        roc_out_thresh = -2.8 / MMOLL_MGDL
        roc_ratio_thresh = 1.3
    elif method == "trial":
        roc_in_thresh = -2.0 / MMOLL_MGDL
        roc_out_thresh = -3.1 / MMOLL_MGDL
        roc_ratio_thresh = 1.5
    elif method == "aggressive":
        roc_in_thresh = -1.9 / MMOLL_MGDL
        roc_out_thresh = -2.8 / MMOLL_MGDL
        roc_ratio_thresh = 1.2
    elif method == "all":
        aggressive_indices, num_aggressive = get_pisa_indices(df.copy(), method="aggressive")
        trial_indices, num_trial = get_pisa_indices(df.copy(), method="trial")
        nominal_indices, num_nominal = get_pisa_indices(df.copy(), method="nominal")
        cautious_indices, num_cautious = get_pisa_indices(df.copy(), method="cautious")
        indices_dict = {'aggressive': pd.Series(aggressive_indices), 'trial': pd.Series(trial_indices), 'nominal': pd.Series(nominal_indices), 'cautious': pd.Series(cautious_indices)}
        pisa_indices = pd.DataFrame(indices_dict)
        num_pisa = num_aggressive
        return pisa_indices, num_pisa

    roc_ratio_min = 0.7
    roc_ratio_max = 1.3
    max_pisa_duration = "90min"
    max_dropout_duration = "60min"

    df["condition_a"] = True
    df["condition_a"].iloc[:2] = False
    df["time_diff_minutes"] = (df.datetime - df.datetime.shift(1)).dt.total_seconds() / 60.0
    df["glucose_value_roc"] = (df.glucose_value - df.glucose_value.shift(1)) / df.time_diff_minutes
    df["glucose_value_acc"] = (df.glucose_value_roc - df.glucose_value_roc.shift(1)) / df.time_diff_minutes
    df["condition_b"] = df.glucose_value_roc < roc_in_thresh
    df["condition_c"] = (df.glucose_value_roc / df.glucose_value_roc.shift(1) > roc_ratio_thresh) | (
                df.glucose_value_roc.shift(1) > 0)

    df["pisa_start"] = (df.condition_a & df.condition_b & df.condition_c).fillna(False)
    if not df.pisa_start.any():
        return [], 0

    df["last_pisa_start_date"] = np.nan
    df.loc[df.pisa_start, "last_pisa_start_date"] = df.loc[df.pisa_start, "datetime"]
    df["last_pisa_start_date"] = df.last_pisa_start_date.ffill().bfill()
    df["condition_d"] = (df.datetime.shift(1).bfill() - df.last_pisa_start_date) >= max_pisa_duration
    df["condition_e"] = (df.datetime - df.datetime.shift(1).bfill()) >= max_dropout_duration
    df["condition_f_0"] = df.glucose_value_roc > roc_out_thresh
    df["condition_f_1"] = False
    df["last_pisa_start_index"] = np.nan
    df.loc[df.pisa_start, "last_pisa_start_index"] = df.index[df.pisa_start].tolist()
    df["last_pisa_start_index"] = df.last_pisa_start_index.ffill()
    df["nth_pisa_entry"] = df.index.tolist() - df.last_pisa_start_index
    df["condition_f_2"] = (df.nth_pisa_entry > 3) & (df.glucose_value_acc < 0) & (df.glucose_value_acc.shift(1) < 0)
    df["condition_f_3"] = (roc_ratio_min <= df.glucose_value_roc.shift(3) / df.glucose_value_roc.shift(1)) & \
                          (df.glucose_value_roc.shift(3) / df.glucose_value_roc.shift(1) <= roc_ratio_max) & \
                          (roc_ratio_min <= df.glucose_value_roc.shift(3) / df.glucose_value_roc) & \
                          (df.glucose_value_roc.shift(3) / df.glucose_value_roc <= roc_ratio_max)
    df["condition_f_4"] = (roc_ratio_min <= df.glucose_value_roc.shift(4) / df.glucose_value_roc.shift(1)) & \
                          (df.glucose_value_roc.shift(4) / df.glucose_value_roc.shift(1) <= roc_ratio_max) & \
                          (roc_ratio_min <= df.glucose_value_roc.shift(4) / df.glucose_value_roc) & \
                          (df.glucose_value_roc.shift(4) / df.glucose_value_roc <= roc_ratio_max)

    df["condition_f"] = df.condition_f_0 & (df.condition_f_1 | df.condition_f_2 | df.condition_f_3 | df.condition_f_4)
    df["pisa_end"] = df.condition_d | df.condition_e | df.condition_f

    df["pisa"] = df.pisa_start * 1 - df.pisa_end * 1
    df["pisa"] = df.pisa.replace(0, np.nan)
    df["pisa"] = df.pisa.ffill().fillna(0)
    df["pisa"] = df.pisa == 1

    pisa_indices = df.loc[df.pisa].index.to_list()
    num_pisa = len(pisa_indices)

    return pisa_indices, num_pisa


def calculate_data_cleaning_kpis(files_df):
    """
    Calculate KPIs across all entries of the passed (excerpt of the) files_df.
    """
    obs_minutes_pre = (files_df["num_measurements_pre"] * files_df["sensor"].str.upper().map(SENSOR_INTERVAL)).sum()
    obs_minutes_post = (files_df["num_measurements_post"] * files_df["sensor"].str.upper().map(SENSOR_INTERVAL)).sum()
    kpis = {"num_files_pre": len(files_df),
            "num_files_post": len(files_df),
            "num_files_dropped": len(files_df) - len(files_df),

            "num_meas_pre": files_df["num_measurements_pre"].sum(),
            "num_meas_post": files_df["num_measurements_post"].sum(),
            "num_meas_dropped": files_df["num_measurements_pre"].sum() - files_df["num_measurements_post"].sum(),

            "obs_days_pre": datetime.timedelta(minutes=int(obs_minutes_pre)).days,
            "obs_days_post": datetime.timedelta(minutes=int(obs_minutes_post)).days,
            "obs_days_dropped": datetime.timedelta(minutes=int(obs_minutes_pre)).days - datetime.timedelta(minutes=int(obs_minutes_post)).days,

            "num_hypo_pre": files_df["num_hypoglycemia_measurements_pre"].sum(),
            "num_hypo_post": files_df["num_hypoglycemia_measurements_post"].sum(),
            "num_hypo_dropped": files_df["num_hypoglycemia_measurements_pre"].sum() - files_df["num_hypoglycemia_measurements_post"].sum(),
            "num_hyper_pre": files_df["num_hyperglycemia_measurements_pre"].sum(),
            "num_hyper_post": files_df["num_hyperglycemia_measurements_post"].sum(),
            "num_hyper_dropped": files_df["num_hyperglycemia_measurements_pre"].sum() - files_df["num_hyperglycemia_measurements_post"].sum(),

            "num_duplicates_pre": files_df["num_duplicates_pre"].sum(),
            "num_duplicates_post": files_df["num_duplicates_post"].sum(),
            "num_duplicates_dropped": files_df["num_duplicates_dropped"].sum(),

            "num_scan_pre": files_df["num_scans_pre"].sum(),
            "num_scan_post": files_df["num_scans_post"].sum(),
            "num_scan_dropped": files_df["num_scans_dropped"].sum(),

            "num_jumps_pre": files_df["num_backwards_jumps_pre"].sum(),
            "num_jumps_post": files_df["num_backwards_jumps_post"].sum(),
            "num_jumps_dropped": files_df["num_backwards_jumps_dropped"].sum(),
            "num_jump_entries_pre": files_df["num_backwards_jump_entries_pre"].sum(),
            "num_jump_entries_post": files_df["num_backwards_jump_entries_post"].sum(),
            "num_jump_entries_dropped": files_df["num_backwards_jump_entries_dropped"].sum(),

            "num_high_dens_pre": files_df["num_high_density_timestamps_pre"].sum(),
            "num_high_dens_post": files_df["num_high_density_timestamps_post"].sum(),
            "num_high_dens_dropped": files_df["num_high_density_timestamps_dropped"].sum(),

            "num_dupl_ts_pre": files_df["num_duplicate_timestamps_pre"].sum(),
            "num_dupl_ts_post": files_df["num_duplicate_timestamps_post"].sum(),
            "num_dupl_ts_dropped": files_df["num_duplicate_timestamps_dropped"].sum(),

            "num_pisa_pre": files_df["num_pisa_pre"].sum(),
            "num_pisa_post": files_df["num_pisa_post"].sum(),
            "num_pisa_dropped": files_df["num_pisa_dropped"].sum()
            }
    return kpis


def generate_output_table(kpis, method, kpis_all=None):
    if method == 'all_sensors':
        table = \
            [
                ["Number of files",
                 kpis['num_files_pre'],
                 f"{kpis['num_files_post']} " + p(kpis['num_files_post'] / kpis['num_files_pre']),
                 f"{kpis['num_files_dropped']} " + p(kpis['num_files_dropped'] / kpis['num_files_pre'])],
                ["Number of measurements",
                 kpis['num_meas_pre'],
                 f"{kpis['num_meas_post']} " + p(kpis['num_meas_post'] / kpis['num_meas_pre']),
                 f"{kpis['num_meas_dropped']} " + p(kpis['num_meas_dropped'] / kpis['num_meas_pre'])],
                ["Observation timespan",
                 f"{kpis['obs_days_pre']} days",
                 f"{kpis['obs_days_post']} days",
                 f"{kpis['obs_days_dropped']} days"],
                ["Measurements in hypoglycemia",
                 f"{kpis['num_hypo_pre']} " + p(kpis['num_hypo_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_hypo_post']} " + p(kpis['num_hypo_post'] / kpis['num_meas_post']),
                 f"{kpis['num_hypo_dropped']} " + p(kpis['num_hypo_dropped'] / kpis['num_meas_pre'])],
                ["Measurements in hyperglycemia",
                 f"{kpis['num_hyper_pre']} " + p(kpis['num_hyper_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_hyper_post']} " + p(kpis['num_hyper_post'] / kpis['num_meas_post']),
                 f"{kpis['num_hyper_dropped']} " + p(kpis['num_hyper_dropped'] / kpis['num_meas_pre'])],
                ["Scan measurements",
                 f"{kpis['num_scan_pre']} " + p(kpis['num_scan_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_scan_post']} " + p(kpis['num_scan_post'] / kpis['num_meas_post']),
                 f"{kpis['num_scan_dropped']} " + p(kpis['num_scan_dropped'] / kpis['num_meas_pre'])],
                ["Backwards jumps in time",
                 f"{kpis['num_jumps_pre']}",
                 f"{kpis['num_jumps_post']}",
                 f"{kpis['num_jumps_dropped']}"],
                ["Backwards jump entries",
                 f"{kpis['num_jump_entries_pre']} " + p(kpis['num_jump_entries_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_jump_entries_post']} " + p(kpis['num_jump_entries_post'] / kpis['num_meas_post']),
                 f"{kpis['num_jump_entries_dropped']} " + p(
                     kpis['num_jump_entries_dropped'] / kpis['num_meas_pre'])],
                ["High density timestamps",
                 f"{kpis['num_high_dens_pre']} " + p(kpis['num_high_dens_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_high_dens_post']} " + p(kpis['num_high_dens_post'] / kpis['num_meas_post']),
                 f"{kpis['num_high_dens_dropped']} " + p(kpis['num_high_dens_dropped'] / kpis['num_meas_pre'])],
                ["Duplicate timestamps",
                 f"{kpis['num_dupl_ts_pre']} " + p(kpis['num_dupl_ts_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_dupl_ts_post']} " + p(kpis['num_dupl_ts_post'] / kpis['num_meas_post']),
                 f"{kpis['num_dupl_ts_dropped']} " + p(kpis['num_dupl_ts_dropped'] / kpis['num_meas_pre'])],
                ["PISAs",
                 f"{kpis['num_pisa_pre']} " + p(kpis['num_pisa_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_pisa_post']} " + p(kpis['num_pisa_post'] / kpis['num_meas_post']),
                 f"{kpis['num_pisa_dropped']} " + p(kpis['num_pisa_dropped'] / kpis['num_meas_pre'])]
            ]
        output_table = tabulate(table, headers=["", "before cleaning", "after cleaning", "dropped"])
    elif method == 'one_sensor':
        table = \
            [
                ["Number of files",
                 f"{kpis['num_files_pre']} [{round(100 * kpis['num_files_pre'] / kpis_all['num_files_pre'], 1)}%]",
                 f"{kpis['num_files_post']} " + p(kpis['num_files_post'] / kpis['num_files_pre']),
                 f"{kpis['num_files_dropped']} " + p(kpis['num_files_dropped'] / kpis['num_files_pre'])],
                ["Number of measurements",
                 f"{kpis['num_meas_pre']} [{round(100 * kpis['num_meas_pre'] / kpis_all['num_meas_pre'], 1)}%]",
                 f"{kpis['num_meas_post']} " + p(kpis['num_meas_post'] / kpis['num_meas_pre']),
                 f"{kpis['num_meas_dropped']} " + p(kpis['num_meas_dropped'] / kpis['num_meas_pre'])],
                ["Observation timespan",
                 f"{kpis['obs_days_pre']} days [{round(100 * kpis['obs_days_pre'] / kpis_all['obs_days_pre'], 1)}%]",
                 f"{kpis['obs_days_post']} days [{100 * round(kpis['obs_days_post'] / kpis_all['obs_days_post'], 1)}%]",
                 f"{kpis['obs_days_dropped']} days " + p(kpis['obs_days_dropped'] / kpis['obs_days_pre'])],
                ["Measurements in hypoglycemia",
                 f"{kpis['num_hypo_pre']} " + p(kpis['num_hypo_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_hypo_post']} " + p(kpis['num_hypo_post'] / kpis['num_meas_post']),
                 f"{kpis['num_hypo_dropped']} " + p(kpis['num_hypo_dropped'] / kpis['num_meas_pre'])],
                ["Measurements in hyperglycemia",
                 f"{kpis['num_hyper_pre']} " + p(kpis['num_hyper_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_hyper_post']} " + p(kpis['num_hyper_post'] / kpis['num_meas_post']),
                 f"{kpis['num_hyper_dropped']} " + p(kpis['num_hyper_dropped'] / kpis['num_meas_pre'])],
                ["Scan measurements",
                 f"{kpis['num_scan_pre']} " + p(kpis['num_scan_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_scan_post']} " + p(kpis['num_scan_post'] / kpis['num_meas_post']),
                 f"{kpis['num_scan_dropped']} " + p(kpis['num_scan_dropped'] / kpis['num_meas_pre'])],
                ["Backwards jumps in time",
                 f"{kpis['num_jumps_pre']}",
                 f"{kpis['num_jumps_post']}",
                 f"{kpis['num_jumps_dropped']}"],
                ["Backwards jump entries",
                 f"{kpis['num_jump_entries_pre']} " + p(kpis['num_jump_entries_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_jump_entries_post']} " + p(kpis['num_jump_entries_post'] / kpis['num_meas_post']),
                 f"{kpis['num_jump_entries_dropped']} " + p(
                     kpis['num_jump_entries_dropped'] / kpis['num_meas_pre'])],
                ["High density timestamps",
                 f"{kpis['num_high_dens_pre']} " + p(kpis['num_high_dens_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_high_dens_post']} " + p(kpis['num_high_dens_post'] / kpis['num_meas_post']),
                 f"{kpis['num_high_dens_dropped']} " + p(kpis['num_high_dens_dropped'] / kpis['num_meas_pre'])],
                ["Duplicate timestamps",
                 f"{kpis['num_dupl_ts_pre']} " + p(kpis['num_dupl_ts_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_dupl_ts_post']} " + p(kpis['num_dupl_ts_post'] / kpis['num_meas_post']),
                 f"{kpis['num_dupl_ts_dropped']} " + p(kpis['num_dupl_ts_dropped'] / kpis['num_meas_pre'])],
                ["PISAs",
                 f"{kpis['num_pisa_pre']} " + p(kpis['num_pisa_pre'] / kpis['num_meas_pre']),
                 f"{kpis['num_pisa_post']} " + p(kpis['num_pisa_post'] / kpis['num_meas_post']),
                 f"{kpis['num_pisa_dropped']} " + p(kpis['num_pisa_dropped'] / kpis['num_meas_pre'])]
            ]
        output_table = tabulate(table, headers=["", "before cleaning\n(rel sens) [rel tot]", "after cleaning\n(rel sens) [rel tot]", "dropped"])

    return output_table


def p(number):
    """
    Convert the number to percent, round to one decimal and convert to string with "()%".
    :param number: number between 0 and 1 to convert
    :return: string: number in percent and rounded to one decimal
    """
    return f"({round(100*number, 1)}%)"


def create_poc_dataset(save_to='3_poc_data/'):
    files = pd.read_excel(DATA_DIR + '3_clean_data/files_info.xlsx')
    files = files.loc[files.study.notna() & files.sensor.notna() & files.treatment.notna()]

    studies = list(files.study.unique())
    sensors = list(files.sensor.unique())
    treatments = list(files.treatment.unique())

    # For every combination of study, sensor and treatment select one file
    filenames = []
    for treatment in treatments:
        for study in studies:
            for sensor in sensors:
                pool_filenames = list(files.loc[(files.study == study) &
                                                (files.sensor == sensor) &
                                                (files.treatment == treatment), 'filename'])
                if len(pool_filenames) > 0:
                    filenames.append(random.choice(pool_filenames))

    cluttered_filenames = filenames
    cluttered_files = files.loc[files.filename.isin(cluttered_filenames)]

    # Prune to have a minimal number of files while still representing every study, sensor and treatment
    for index, file in cluttered_files.iterrows():
        cluttered_files_wo_file = cluttered_files[cluttered_files.file_id != file.file_id]
        remaining_studies = list(cluttered_files_wo_file.study.unique())
        remaining_sensors = list(cluttered_files_wo_file.sensor.unique())
        remaining_treatments = list(cluttered_files_wo_file.treatment.unique())
        if file.study in remaining_studies and file.sensor in remaining_sensors and file.treatment in remaining_treatments:
            filenames.remove(file.filename)
            cluttered_files.drop(file.file_id, inplace=True)

    # Delete existing POC data, re-create POC folder and save Excel with file information
    # shutil.rmtree(DATA_DIR + save_to)
    os.makedirs(DATA_DIR + save_to)
    poc_files = files.loc[files.filename.isin(filenames)]
    poc_files.to_excel(DATA_DIR + save_to + 'files_info.xlsx', index=False)

    # Copy files to POC folder
    for filename in filenames:
        original = DATA_DIR + '3_clean_data/' + filename
        target = DATA_DIR + '3_poc_data/' + filename
        shutil.copyfile(original, target)
