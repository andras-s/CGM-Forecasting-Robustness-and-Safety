import pandas as pd
from utils.setup import DATA_DIR

files = pd.read_excel(DATA_DIR + '3_clean_data/files_info.xlsx')
num_files = len(files)


treatments = {
    'basal': 'BI',
    'basal only': 'BI',
    'insulin+nia': 'BI',
    'MDI': 'MDI',
    'CSII': 'CSII',
    'Medtronic 640G': 'CSII',
    'HYBRID_mm670g': 'AP',
    'HYBRID_mm780g': 'AP',
    'HYBRID_accucheck': 'AP',
}
files = files.replace({'treatment': treatments})


print('Sex:   ', files.sex.value_counts() * 100/num_files)
print('Age:   ', files.age.value_counts(bins = [0, 25, 50, 75, 100]) * 100/num_files)
print(f'Age: {files.age.min()} - {files.age.max()}')
print('Diabetes type:   ', files.diabetes_type.value_counts() * 100/num_files)
print('Treatment:   ', files.treatment.value_counts() * 100/num_files)
print('HbA1c:   ', files.hba1c.value_counts(bins = [0.000, 6.000, 6.999, 8.499, 10.499, 100]) * 100/num_files)
print('Sensor:   ', files.sensor.value_counts() * 100/num_files)
# concat the right columns
# count occurrences


