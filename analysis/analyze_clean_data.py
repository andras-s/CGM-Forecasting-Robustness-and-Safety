import pandas as pd
from utils.data.load import read_data
from utils.setup import DATA_DIR


load_dir = '3_clean_data/'
file_id = 146


files_info = pd.read_excel(DATA_DIR + load_dir + 'files_info.xlsx')
data, file_ids = read_data(load_dir=load_dir)

file_info = files_info.loc[files_info.file_id == file_id]
with pd.option_context("display.max_columns", None):
    with pd.option_context('display.expand_frame_repr', False):
        print(file_info)

file_position = file_ids.index(str(file_id))
file_df = data[file_position]
file_df.datetime = pd.to_datetime(file_df.datetime)
file_df['datetime_diff'] = file_df.datetime.diff()


end = 0
