import socket
import os
import torch
import pandas as pd
from pathlib import Path


# Defining global constants that can be used within other files
DOMAINS = dict(
    diabetes_type = {
        'DM1': ['1'],
        'DM2': ['2'],
        # 'GDM': ['GDM'],
        # 'MODY': ['MODY'],
        # 'other': ['other'],
        'other': ['other', 'pancreatogenic', 'GDM', 'GIDM', 'MIDD', 'CF-related', 'MODY', 'posttransplant'],
    },
    gender ={
        'male': ['m'],
        'female': ['f']
    },
    treatment = {
        'BI': ['basal', 'basal only', 'insulin+nia'],
        'MDI': ['MDI'],
        'CSII': ['CSII', 'Medtronic 640G'],
        'AP': ['HYBRID_mm670g', 'HYBRID_mm780g', 'HYBRID_accucheck'],
    },
    sensor = {
        'libre': ['libre'],
        'dexcom': ['dexcom'],
        'medtronic': ['medtronic']
    },
    HbA1c_level = {
        'Non-Diabetic': [0.000, 6.000],
        'In Control': [6.001, 6.999],
        'Monitor Closely': [7.000, 8.499],
        'Elevated': [8.500, 10.499],
        'Seriously Elevated': [10.500, 100]
    },
    glucose_level = {
        'low': [0, 70 / 18.02],
        'in range': [70 / 18.02, 180 / 18.02],
        'high': [180 / 18.02, 100]
    },
    slope = {
        'rising': [0.055494, 10],           # calculated to be 1/3 quantile of all slopes
        'stable': [-0.073992, 0.055494],    # calculated to be 1/3 quantile of all slopes
        'falling': [-10, -0.073992]         # calculated to be 1/3 quantile of all slopes
    },
)

# These constants were used to standardise the file location
# They need to be adapted for other users

ROOT_DIR = Path(__file__).parent.parent

if socket.gethostname() == 'mtec-im-gpu01':
    DATA_DIR = "/local/home/ansass/Thesis/icarus/analysis/data_on_server/v_2/"
elif socket.gethostname() == 'mtec-mis-502':
    DATA_DIR = os.path.expanduser("/mnt/wave/icarus/data/")
else:
    DATA_DIR = "C:/Users/sass9/PycharmProjects/Thesis/Data Backup/data_on_server/v_2/"
    print("You are running the code on an undefined machine. Please add your machine and data paths to utils/setup.py")


def init_gpu(gpu):
    """
    Initialises a GPU for pytorch experiments
    """
    if torch.cuda.is_available() and gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        device = torch.device(int(gpu))
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        device_name = 'cpu'

    print("Using device: " + device_name)

    return device


def load_patient_info():
    patient_info = pd.read_excel(DATA_DIR + '/andras/patients.xlsx', engine='openpyxl')
    patient_info.rename(columns={'Code (med=medtronic; dex=dexcom; fs=freestyle libre)': 'code',
                                    'ID': 'patient',
                                    'device (1=medtronic, 2=dexcom, 3=libre)': 'sensor',
                                    'Geschlecht (1=m, 2=f)': 'gender',
                                    'Type Diabetes (1=DM1, 2=DM2, 3=GDM, 4=MODY, 5=other)': 'diabetes_type',
                                    'Therapy (1=MDI, 2=CSII, 3=no insulin antidiabetics [nia], 4=basal insulin, 5=insulin & nia, 6=other)': 'therapy'},
                           inplace=True)
    patient_info = patient_info[['patient', 'sensor', 'gender', 'diabetes_type']]  # 'therapy' has multiple entries for the same patient
    patient_info['sensor'].replace({1: 'medtronic', 2: 'dexcom', 3: 'libre'}, inplace=True)
    patient_info['gender'].replace({1: 'm', 2: 'f'}, inplace=True)
    patient_info['diabetes_type'].replace({1: 'DM1', 2: 'DM2', 3: 'GDM', 4: 'MODY', 5: 'other', 6: 'other', 99: 'other'}, inplace=True)
    # patient_data_df['therapy'].replace({1: 'MDI', 2: 'CSII', 3: 'nia', 4: 'basal insulin', 5: 'insulin+nia', 6: 'other', 99: 'other'}, inplace=True)
    patient_info.drop_duplicates(inplace=True)
    return patient_info
