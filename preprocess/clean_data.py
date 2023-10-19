import os
import pandas as pd
from natsort import natsorted

from utils.calc import MMOLL_MGDL
from utils.setup import DATA_DIR
from preprocess.helper_functions import create_files_df, get_jump_indices, get_high_density_indices, \
    get_pisa_indices, calculate_data_cleaning_kpis, generate_output_table, p, create_poc_dataset
from utils.plot import create_plots_around_indices, create_pisa_plots



class DataCleaner:
    def __init__(self,
                 studies=None,
                 sensors=None,
                 file_formats=None,
                 file_ids=None,
                 backwards_jump_method="drop",
                 pisa_method="nominal",
                 save_concat_dir='1_concatenated_data/',
                 save_wrangle_dir='2_wrangled_data/',
                 save_dir='3_clean_data/',
                 save_intermediate_data=True,
                 plot_scan_regions=False,
                 plot_backward_datetime_jump_regions=False,
                 plot_high_density_regions=False,
                 plot_duplicate_timestamp_regions=False,
                 plot_pisa_regions=False
                 ):
        """
        Loads the raw data of the specified patients and performs multiple cleaning steps.
            1. The raw data of each patient is loaded and concatenated (might be saved in multiple CSV files)
            2. The data coming from different sensors have different column names, orientation and units. The data is
            brought into a standard form.
            3. Exact duplicate measurements (same datetime and glucose value) are dropped
            4. In some cases there are backwards datetime jumps in the data (e.g. due to change from summer to winter
            time, travelling east, or correction of date and/or time). These cause overlapping time series which have to
            be dealt with. Data coming from the sensors from medtronic or dexcom has a natural ordering, so such jumps
            can be identified easily. In these cases the overlapping regions are dropped. For the other sensors these
            issues are dealt with in the next step.
            5. In the data there are regions with more measurements than expected. These regions are the result of many
            different reasons, one being the same as in 4. for the remaining sensors. These regions are identified are
            dropped.
            6. Remaining timestamps which appear twice are dropped in this step.
            7. As a final step entries affected by PISA (pressure-induced sensor attenuation) are identified by an
            algorithm and dropped.
        The cleaned Dataframes (1 per patient) are saved as CSVs.
        :param sensors: (list) Only the raw data of patients with the specified sensors is cleaned.
        :param patient_ids: (list) Only the raw data of patients with the specified IDs is cleaned.
        :param backwards_jump_method: Two methods are available to handle backwards datetime jumps. Either the entries
        are "drop"-ped or for each of the ~65 jumps "subpatient"-s are introduced and separate CSVs are produced.
        :param pisa_method: Different settings for PISA entry removal are implemented. They differ in aggressivity
        (more aggressive = more entries are removed). Options from least to most aggressive: "cautious", "nominal",
        "trial".
        :param save_concat_dir: filepath to intermediate results.
        :param save_wrangle_dir: filepath to intermediate results.
        :param save_dir: filepath to final CSVs.
        :param save_intermediate_data: boolean determining whether intermediate results should be saved.
        :param plot_high_density_regions: boolean determining whether regions with high density timestamps should be plotted.
        :param plot_duplicate_timestamp_regions: boolean determining whether regions with duplicate timestamps should be plotted.
        :param plot_pisa_regions: boolean determining whether regions with PISA entries should be plotted.
        """
        self.backwards_jump_method = backwards_jump_method
        self.pisa_method = pisa_method
        self.save_concat_dir = save_concat_dir
        self.save_wrangle_dir = save_wrangle_dir
        self.save_dir = save_dir
        self.save_path = DATA_DIR + save_dir
        self.save_intermediate_data = save_intermediate_data
        self.plot_scan_regions = plot_scan_regions
        self.plot_backward_datetime_jump_regions = plot_backward_datetime_jump_regions
        self.plot_high_density_regions = plot_high_density_regions
        self.plot_duplicate_timestamp_regions = plot_duplicate_timestamp_regions
        self.plot_pisa_regions = plot_pisa_regions

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.files = create_files_df(save_to=self.save_path)

        if studies is None:
            studies = ["wca", "hybrid1", "hybrid2", "dia_hack"]
        self.studies = studies

        if sensors is None:
            sensors = ["libre", "medtronic", "dexcom"]
        self.sensors = sensors

        if file_formats is None:
            file_formats = list(self.files.raw_file_format.unique())
        self.file_formats = file_formats

        if file_ids is None:
            file_ids = list(self.files.file_id.unique())
        self.file_ids = file_ids

        print("Cleaning files...")
        for index, file in self.files.iterrows():
            if file.study in self.studies and \
               file.sensor in self.sensors and \
               file.raw_file_format in self.file_formats and \
               file.file_id in self.file_ids:
                print(f"\n----------   File {file.file_id}   {file.study}   {file.patient_id_study}   {file.sensor}   ----------")
                df = self.concatenate_patient_data(file)
                df = self.wrangle_data(df, file)
                df = self.handle_duplicates(df, index)
                df = self.handle_scans(df, index, file)
                df = self.handle_backwards_datetime_jumps(df, index, file)
                df = self.handle_high_density_regions(df, index, file)
                df = self.handle_duplicate_timestamps(df, index, file)
                df = self.handle_pisa(df, index, file)
                df.to_csv(self.save_path + file.filename, index=False)
                self.print_data_cleaning_report(df, index)

        self.files.to_excel(self.save_path + 'files_info.xlsx', index=False)
        create_poc_dataset(save_to='3_poc_data/')
        self.print_data_cleaning_report()

    def concatenate_patient_data(self, file):
        """
        Given a patients sensor and files the concatenated file is created
        """
        study = file.study
        file_format = file.raw_file_format
        raw_files = file.wca_raw_filenames
        filename = file.filename

        if study == 'wca':
            if file_format == "medtronic":
                raw_files = natsorted(raw_files, reverse=True)
                df = pd.concat(
                    [pd.read_excel(DATA_DIR + "/0_raw_data/WCA/raw/" + f, engine='openpyxl', skiprows=1) for f in
                     raw_files],
                    ignore_index=True)
            elif file_format == "dexcom":
                raw_files = natsorted(raw_files)
                df = pd.concat(
                    [pd.read_excel(DATA_DIR + "/0_raw_data/WCA/raw/" + f, engine='openpyxl', skiprows=1) for f in
                     raw_files],
                    ignore_index=True)
                df = df.sort_values(by='GlucoseInternalTime')
            elif file_format == "libre":
                raw_files = natsorted(raw_files)
                df = pd.concat(
                    [pd.read_excel(DATA_DIR + "/0_raw_data/WCA/raw/" + f, engine='openpyxl', skiprows=1) for f in
                     raw_files],
                    ignore_index=True)
        else:
            return None

        if self.save_intermediate_data:
            if not os.path.exists(DATA_DIR + self.save_concat_dir):
                os.makedirs(DATA_DIR + self.save_concat_dir)
            df.to_csv(DATA_DIR + self.save_concat_dir + filename, index=False)
        return df

    def wrangle_data(self, df, file):
        """
        Converts raw CGM data to standard format across CGM sensor types.
        """
        study = file.study
        file_format = file.raw_file_format
        filename = file.filename
        raw_filename = file.raw_filename

        if study == 'wca':
            if file_format == 'medtronic':
                df = df.loc[df['Time'].astype(str).str.len() == 8]
                df = df.dropna(subset=['Sensor Glucose (mmol/L)']).copy()
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                df = df.rename(columns={'Sensor Glucose (mmol/L)': 'glucose_value'})
                df = df[['datetime', 'glucose_value']][::-1].reset_index(drop=True)
            elif file_format == 'dexcom':
                df['datetime'] = pd.to_datetime(df['GlucoseDisplayTime'])
                df['internal_time'] = pd.to_datetime(df['GlucoseInternalTime'])
                df = df.sort_values(by='internal_time')
                df['glucose_value'] = df['GlucoseValue'].replace({'Hoch': None, 'Niedrig': None})
                df = df[['datetime', 'internal_time', 'glucose_value']]
            elif file_format == 'libre':
                df = df[df['Time'].astype(str).str.len() == 8].copy()
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                df['glucose_value'] = df['Value'] / MMOLL_MGDL
            if file_format == 'libre':
                df['Comment'] = df['Comment'].str.rstrip()
                df['Comment'] = df['Comment'].replace({'NOSCAN^NEW SENSOR': 'NOSCAN'})
                df['scan'] = df['Comment'] != 'NOSCAN'
                df.loc[df['Comment'].isnull(), 'scan'] = False
                df = df[['datetime', 'glucose_value', 'scan']].reset_index(drop=True)
            else:
                df = df[['datetime', 'glucose_value']]

        if study == 'hybrid1':
            filepath = DATA_DIR + '/0_raw_data/HYBRID1/proc/' + raw_filename
            # (note the dexcom file is actually not dexcom but FSL, is probably a mistake)
            if file_format == 'FSL' or file_format == 'DEXCOM':
                df = pd.read_csv(filepath, sep=';', header=1)
                df['datetime'] = (df['Date'] + ' ' + df['Time']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M')
                df['glucose_value'] = df['Value'] / MMOLL_MGDL
            elif file_format == 'MM640G' or file_format == 'MM670G':
                df = pd.read_csv(filepath, sep=';', header=1)
                df['datetime'] = (df['Date'] + ' ' + df['Time']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/L)': 'glucose_value'})
                df = df[::-1].reset_index(drop=True)
            # Handle scanning in libre and FSL sensors
            if file_format == 'FSL':
                df['Comment'] = df['Comment'].str.rstrip()
                df['Comment'] = df['Comment'].replace({'NOSCAN^NEW SENSOR': 'NOSCAN'})
                df['scan'] = df['Comment'] != 'NOSCAN'
                df.loc[df['Comment'].isnull(), 'scan'] = False
                df = df[['datetime', 'glucose_value', 'scan']].reset_index(drop=True)
            else:
                df = df[['datetime', 'glucose_value']]

        if study == 'hybrid2':
            filepath = DATA_DIR + '/0_raw_data/HYBRID2/proc/' + raw_filename
            if file_format == 'MM640G' or file_format == 'MM670G' or file_format == 'MM780G':
                df = pd.read_csv(filepath, sep=';', header=1)
                df['datetime'] = (df['Date'] + ' ' + df['Time']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/L)': 'glucose_value'})
                df = df[::-1].reset_index(drop=True)
            elif file_format == 'dexcom':
                df = pd.read_csv(filepath, sep=';', header=1, encoding='ISO-8859-1')
                df['datetime'] = (df['Date'] + ' ' + df['Time']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Glukosewert (mmol/l)': 'glucose_value'})
                df = df.loc[(df.glucose_value != 'Hoch') & (df.glucose_value != 'Niedrig')].reset_index(drop=True)
            elif file_format == 'dexcom_MDI_111':
                df = pd.read_csv(filepath, sep=';', header=1, encoding='ISO-8859-1')
                df['datetime'] = (df['Date'] + ' ' + df['Time ']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Glukosewert (mmol/l)': 'glucose_value'})
                df = df.loc[(df.glucose_value != 'Hoch') & (df.glucose_value != 'Niedrig')].reset_index(drop=True)
            elif file_format == 'libre_dexcom_mmparadigm':
                df = pd.read_csv(filepath, sep=';', header=1, encoding='ISO-8859-1')
                df['datetime'] = (df['Date'] + ' ' + df['Time']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/l)': 'glucose_value'}).reset_index(drop=True)
            elif file_format == 'libre_dexcom':
                df = pd.read_csv(filepath, sep=';', header=1, encoding='ISO-8859-1')
                df['datetime'] = (df['Date'] + ' ' + df['Zeit']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/l)': 'glucose_value'}).reset_index(drop=True)
            elif file_format == 'libre_mm640g':
                df = pd.read_csv(filepath, sep=';', header=1, encoding='ISO-8859-1')
                df['datetime'] = (df['Datum'] + ' ' + df['Uhrzeit']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/l)': 'glucose_value'}).reset_index(drop=True)
            elif file_format == 'libre_accucheck':
                df = pd.read_csv(filepath, sep=';', header=1, encoding='ISO-8859-1')
                df['datetime'] = (df['Datum'] + ' ' + df['Zeit']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/l)': 'glucose_value'}).reset_index(drop=True)
            df = df[['datetime', 'glucose_value']]

        if study == 'dia_hack':
            filepath = DATA_DIR + '/0_raw_data/DIA-HACK/raw/' + raw_filename
            if file_format == 'dexcom':
                df = pd.read_csv(filepath, sep=',', header=0, skiprows=range(1, 12), encoding='latin-1')
                df = df[df["Zeitstempel (JJJJ-MM-TTThh:mm:ss)"].notna()]
                df['datetime'] = (df['Zeitstempel (JJJJ-MM-TTThh:mm:ss)']).apply(pd.to_datetime, format='%Y-%m-%dT%H:%M:%S')
                df = df.rename(columns={'Glukosewert (mmol/l)': 'glucose_value'})
                df['glucose_value'] = pd.to_numeric(df['glucose_value'], errors='coerce')
                df = df.reset_index(drop=True)
            elif file_format == 'libre':
                df = pd.read_csv(filepath, header=2)
                df = df.loc[df.Aufzeichnungstyp != 6.0]
                colnames = df.columns
                entries = df.Gerät.str.replace('"', '')
                entries = entries.str.split(',', expand=True)
                entries = entries[entries[4] != '']
                entries[4] = entries[4] + '.' + entries[5]
                entries.drop(columns=5, inplace=True)
                entries = entries.loc[:, :len(colnames)]
                entries.columns = colnames
                df = entries
                df['datetime'] = (df['Gerätezeitstempel']).apply(pd.to_datetime, format='%d-%m-%Y %H:%M')
                df = df.rename(columns={'Glukosewert-Verlauf mmol/L': 'glucose_value'})
                df = df.loc[df.Aufzeichnungstyp == '0']
                serial_numbers = df.Seriennummer.unique()
                num_measurements = {}
                max_measurements = -1
                max_serial = None
                for serial in serial_numbers:
                    num_measurements_serial = len(df.loc[df.Seriennummer == serial])
                    num_measurements[serial] = num_measurements_serial
                    if num_measurements_serial > max_measurements:
                        max_measurements = num_measurements_serial
                        max_serial = serial
                df = df.loc[df.Seriennummer == max_serial].reset_index(drop=True)
            elif file_format == 'medtronic':
                df = pd.read_csv(filepath, sep=';', header=6, on_bad_lines='skip')
                cut_index = df.index[df['Sensor Glucose (mmol/L)'] == 'Sensor Glucose (mmol/L)'][-1]
                df = df.loc[cut_index + 1:]
                df['datetime'] = (df['Date'] + ' ' + df['Time']).apply(pd.to_datetime, format='%d.%m.%Y %H:%M:%S')
                df = df.rename(columns={'Sensor Glucose (mmol/L)': 'glucose_value'})
                df['glucose_value'] = pd.to_numeric(df['glucose_value'])
                df = df[['datetime', 'glucose_value']][::-1].reset_index(drop=True)

            df = df[['datetime', 'glucose_value']]

        df['glucose_value'] = df['glucose_value'].astype('float64')
        df = df[df["glucose_value"].notna()]

        if self.save_intermediate_data:
            if not os.path.exists(DATA_DIR + self.save_wrangle_dir):
                os.makedirs(DATA_DIR + self.save_wrangle_dir)
            df.to_csv(DATA_DIR + self.save_wrangle_dir + filename, index=False)

        return df

    def handle_duplicates(self, df, index):
        """
        Duplicate entries (same datetime and glucose value) are dropped and documented.
        """
        num_dupl = df.duplicated().sum()

        cleaned_df = df.drop_duplicates()
        cleaned_num_dupl = df.duplicated().sum()

        self.files.loc[index, "num_measurements_pre"] = len(cleaned_df)
        self.files.loc[index, "num_hypoglycemia_measurements_pre"] = (cleaned_df["glucose_value"] < 3.9).sum()
        self.files.loc[index, "num_hyperglycemia_measurements_pre"] = (cleaned_df["glucose_value"] > 11.1).sum()

        original_num_measurements = self.files.loc[index, 'num_measurements_pre']
        self.files.loc[index, "num_duplicates_pre"] = num_dupl
        self.files.loc[index, "percent_duplicates_pre"] = num_dupl / original_num_measurements

        self.files.loc[index, "num_duplicates_post"] = cleaned_num_dupl
        self.files.loc[index, "percent_duplicates_post"] = cleaned_num_dupl / len(cleaned_df)
        self.files.loc[index, "num_duplicates_dropped"] = len(df) - len(cleaned_df)
        self.files.loc[index, "percent_duplicates_dropped"] = (len(df) - len(cleaned_df)) / original_num_measurements

        return cleaned_df.reset_index(drop=True)

    def handle_scans(self, df, index, file):
        """
        Scan entries in libre and FSL sensors are dropped and documented.
        """
        sensor = file.sensor
        file_id = file.file_id
        study = file.study
        raw_file_format = file.raw_file_format

        cleaned_df = df.copy()
        if study in ['wca', 'hybrid1'] and raw_file_format in ["libre", "FSL"]:
            scan_indices = df[df.scan].index
            num_scans = len(scan_indices)

            original_num_measurements = self.files.loc[index, 'num_measurements_pre']
            self.files.loc[index, "num_scans_pre"] = num_scans
            self.files.loc[index, "percent_scans_pre"] = num_scans / original_num_measurements

            cleaned_df = df[~df['scan']].reset_index(drop=True)
            cleaned_num_scans = len(cleaned_df[cleaned_df.scan])
            cleaned_num_measurements = len(cleaned_df)

            self.files.loc[index, "num_scans_post"] = cleaned_num_scans
            self.files.loc[index, "percent_scans_post"] = cleaned_num_scans / cleaned_num_measurements
            self.files.loc[index, "num_scans_dropped"] = len(df) - cleaned_num_measurements
            self.files.loc[index, "percent_scans_dropped"] = (len(df) - cleaned_num_measurements) / original_num_measurements

            ### Plotting
            if self.plot_scan_regions:
                create_plots_around_indices(df, scan_indices, "scan_regions", sensor, file_id, mark_indices=scan_indices)

        return cleaned_df.reset_index(drop=True)

    def handle_backwards_datetime_jumps(self, df, index, file):
        """
        Determine entries where the datetime is not in order and handle the following entries.
            "drop": drops entries within the overlapping regions
            "subpatients": create for each of the found jumps a subpatient such that no data has to be dropped. Multiple
            CSV files are saved for those patients.
        """
        study = file.study
        file_id = file.file_id

        jump_indices, num_jumps, overlap_positions, num_overlap_entries = get_jump_indices(df.copy())

        original_num_measurements = self.files.loc[index, 'num_measurements_pre']
        self.files.loc[index, "num_backwards_jumps_pre"] = num_jumps
        self.files.loc[index, "num_backwards_jump_entries_pre"] = num_overlap_entries

        cleaned_df = df[~overlap_positions].copy()
        cleaned_jump_indices, cleaned_num_jumps, cleaned_overlap_positions, cleaned_num_overlap_entries = get_jump_indices(cleaned_df.copy())
        len_cleaned_df = len(cleaned_df)

        ### Plotting
        if self.plot_backward_datetime_jump_regions:
            create_plots_around_indices(df, jump_indices, "backward_datetime_jump_regions", study, file_id, mark_indices=df[overlap_positions].index)

        self.files.loc[index, "num_backwards_jumps_post"] = cleaned_num_jumps
        self.files.loc[index, "num_backwards_jumps_dropped"] = num_jumps - cleaned_num_jumps

        self.files.loc[index, "num_backwards_jump_entries_post"] = cleaned_num_overlap_entries
        self.files.loc[index, "percent_backwards_jump_entries_post"] = cleaned_num_overlap_entries / len_cleaned_df
        self.files.loc[index, "num_backwards_jump_entries_dropped"] = len(df) - len_cleaned_df
        self.files.loc[index, "percent_backwards_jump_entries_dropped"] = (len(df) - len_cleaned_df) / original_num_measurements

        return cleaned_df.reset_index(drop=True)

    def handle_high_density_regions(self, df, index, file):
        """
        Remove glucose measurements that occur above the known frequency of the device.
        """
        study = file.study
        file_id = file.file_id
        sensor = file.sensor

        original_num_measurements = self.files.loc[index, 'num_measurements_pre']
        len_dfs = len(df)

        high_density_indices, num_high_density_locations = get_high_density_indices(df, sensor)
        num_high_density_timestamps_pre = num_high_density_locations

        ### Handling of high density regions
        #if study in ['wca', 'hybrid1']:
        #     if file_format == "libre" or file_format == "MM670G" or file_format == "MM640G" or file_format == "DEXCOM":
        #        cleaned_df = df[~df.index.isin(high_density_indices)].reset_index(drop=True).copy()
        #    else:
        #        cleaned_df = df
        cleaned_df = df[~df.index.isin(high_density_indices)].reset_index(drop=True).copy()

        cleaned_high_density_indices, cleaned_num_high_density_locations = get_high_density_indices(cleaned_df, sensor)
        num_high_density_timestamps_post = cleaned_num_high_density_locations
        len_cleaned_df = len(cleaned_df)
        cleaned_df = cleaned_df

        ### Plotting
        if self.plot_high_density_regions:
            create_plots_around_indices(df, high_density_indices, "high_density_regions", study, file_id, mark_indices=high_density_indices)

        self.files.loc[index, "num_high_density_timestamps_pre"] = num_high_density_timestamps_pre
        self.files.loc[index, "percent_high_density_timestamps_pre"] = num_high_density_timestamps_pre / original_num_measurements
        self.files.loc[index, "num_high_density_timestamps_post"] = num_high_density_timestamps_post
        self.files.loc[index, "percent_high_density_timestamps_post"] = num_high_density_timestamps_post / len_cleaned_df
        self.files.loc[index, "num_high_density_timestamps_dropped"] = len_dfs - len_cleaned_df
        self.files.loc[index, "percent_high_density_timestamps_dropped"] = (len_dfs - len_cleaned_df) / original_num_measurements

        return cleaned_df[["datetime", "glucose_value"]]

    def handle_duplicate_timestamps(self, df, index, file):
        """
        Remove remaining duplicate timestamps.
        """
        study = file.study
        file_format = file.raw_file_format
        sensor = file.sensor
        file_id = file.file_id

        original_num_measurements = self.files.loc[index, 'num_measurements_pre']
        len_dfs = len(df)

        duplicate_indices = df[df["datetime"].duplicated(keep=False)].index.to_list()
        num_duplicates = len(duplicate_indices)
        num_duplicates_pre = num_duplicates

        ### Handling of duplicates
        # if study in ['wca', 'hybrid1']:
        #     if file_format == "medtronic" or file_format == "MM670G" or file_format == "MM640G" or file_format == "FSL":
        #         cleaned_df = df.loc[~df.index.isin(duplicate_indices)].reset_index(drop=True).copy()
        #     else:
        #         cleaned_df = df
        cleaned_df = df.loc[~df.index.isin(duplicate_indices)].reset_index(drop=True).copy()

        cleaned_duplicate_indices = cleaned_df[cleaned_df["datetime"].duplicated(keep=False)].index.to_list()
        cleaned_num_duplicates = len(cleaned_duplicate_indices)
        num_duplicates_post = cleaned_num_duplicates
        len_cleaned_df = len(cleaned_df)

        ### Plotting
        if self.plot_duplicate_timestamp_regions:
            create_plots_around_indices(df, duplicate_indices, "duplicate_timestamp_regions", sensor, file_id)

        self.files.loc[index, "num_duplicate_timestamps_pre"] = num_duplicates_pre
        self.files.loc[index, "percent_duplicate_timestamps_pre"] = num_duplicates_pre / original_num_measurements
        self.files.loc[index, "num_duplicate_timestamps_post"] = num_duplicates_post
        self.files.loc[index, "percent_duplicate_timestamps_post"] = num_duplicates_post / len_cleaned_df
        self.files.loc[index, "num_duplicate_timestamps_dropped"] = len_dfs - len_cleaned_df
        self.files.loc[index, "percent_duplicate_timestamps_dropped"] = (len_dfs - len_cleaned_df) / original_num_measurements

        return cleaned_df[["datetime", "glucose_value"]]

    def handle_pisa(self, df, index, file):
        """
        Drop entries affected by PISA (pressure-induced sensor attenuation). Different settings for PISA entry removal
        are implemented. They differ in aggressivity (more aggressive = more entries are removed). Options from least to
        most aggressive: "cautious", "nominal", "trial".
        """
        sensor = file.sensor
        file_id = file.file_id

        original_num_measurements = self.files.loc[index, 'num_measurements_pre']
        len_df = len(df)
        num_pisa_pre = 0

        pisa_indices, num_pisa = get_pisa_indices(df, method=self.pisa_method)
        num_pisa_pre += num_pisa

        cleaned_df = df.loc[~df.index.isin(pisa_indices)]
        cleaned_pisa_indices, cleaned_num_pisa = get_pisa_indices(cleaned_df)

        num_pisa_post = cleaned_num_pisa
        len_cleaned_df = len(cleaned_df)
        cleaned_df = cleaned_df.reset_index(drop=True)

        ### Plotting
        if self.plot_pisa_regions:
            if not (self.pisa_method == 'all'):
                create_plots_around_indices(df, pisa_indices, "pisa_regions", sensor, file_id, mark_indices=pisa_indices)
            elif self.pisa_method == 'all':
                create_pisa_plots(df, pisa_indices, "pisa_regions", sensor, file_id)

        self.files.loc[index, "num_pisa_pre"] = num_pisa_pre
        self.files.loc[index, "percent_pisa_pre"] = num_pisa_pre / original_num_measurements
        self.files.loc[index, "num_pisa_post"] = num_pisa_post
        self.files.loc[index, "percent_pisa_post"] = num_pisa_post / len_cleaned_df
        self.files.loc[index, "num_pisa_dropped"] = len_df - len_cleaned_df
        self.files.loc[index, "percent_pisa_dropped"] = (len_df - len_cleaned_df) / original_num_measurements

        return cleaned_df[["datetime", "glucose_value"]]

    def print_data_cleaning_report(self, df=None, index=None):
        """
        Given a patients ID. The patient's data cleaning report is printed. If no ID is specified, a report is printed
        for the whole dataset of patients.
        """
        if index is not None:
            self.files.loc[index, "num_measurements_post"] = len(df)
            self.files.loc[index, "num_hypoglycemia_measurements_post"] = (df["glucose_value"] < 3.9).sum()
            self.files.loc[index, "num_hyperglycemia_measurements_post"] = (df["glucose_value"] > 11.1).sum()

            num_measurements_pre = self.files.loc[index, 'num_measurements_pre'].item()
            num_measurements_post = self.files.loc[index, "num_measurements_post"].item()
            percent_measurements_post = num_measurements_post / num_measurements_pre

            # num_duplicates = self.patients.loc[index, 'num_duplicates_pre'].item()
            # percent_duplicates = self.patients.loc[index, 'percent_duplicates_pre'].item()
            # num_duplicates_dropped = self.patients.loc[index, 'num_duplicates_dropped'].item()
            # percent_duplicates_dropped = self.patients.loc[index, 'percent_duplicates_dropped'].item()

            num_jumps = self.files.loc[index, 'num_backwards_jumps_pre'].item()
            num_jumps_dropped = self.files.loc[index, 'num_backwards_jump_entries_dropped'].item()
            percent_jumps_dropped = self.files.loc[index, 'percent_backwards_jump_entries_dropped'].item()

            num_high_density_timestamps = self.files.loc[index, 'num_high_density_timestamps_pre'].item()
            percent_high_density_timestamps = self.files.loc[index, 'percent_high_density_timestamps_pre'].item()
            num_high_density_timestamps_dropped = self.files.loc[index, 'num_high_density_timestamps_dropped'].item()
            percent_high_density_timestamps_dropped = self.files.loc[index, 'percent_high_density_timestamps_dropped'].item()

            num_duplicate_timestamps = self.files.loc[index, 'num_duplicate_timestamps_pre'].item()
            percent_duplicate_timestamps = self.files.loc[index, 'percent_duplicate_timestamps_pre'].item()
            num_duplicate_timestamps_dropped = self.files.loc[index, 'num_duplicate_timestamps_dropped'].item()
            percent_duplicate_timestamps_dropped = self.files.loc[index, 'percent_duplicate_timestamps_dropped'].item()

            num_pisa_pre = self.files.loc[index, 'num_pisa_pre'].item()
            percent_pisa_pre = self.files.loc[index, 'percent_pisa_pre'].item()
            num_pisa_dropped = self.files.loc[index, 'num_pisa_dropped'].item()
            percent_pisa_dropped = self.files.loc[index, 'percent_pisa_dropped'].item()

            print(f"Number of measurements before cleaning: {num_measurements_pre}")
            print(f"Number of measurements after cleaning: {num_measurements_post} " + p(percent_measurements_post))
            # print(f"DROPPED: {num_duplicates_dropped} " + p(percent_duplicates_dropped) + f" from {num_duplicates} " + p(
            #     percent_duplicates) + " duplicate measurements")
            print(f"DROPPED: {num_jumps_dropped} " + p(percent_jumps_dropped) + f" entries from {int(num_jumps)} backwards jumps in datetime")
            print(f"DROPPED: {num_high_density_timestamps_dropped} " + p(
                percent_high_density_timestamps_dropped) + f" from {num_high_density_timestamps} " + p(
                percent_high_density_timestamps) + " high density measurements")
            print(f"DROPPED: {num_duplicate_timestamps_dropped} " + p(
                percent_duplicate_timestamps_dropped) + f" from {num_duplicate_timestamps} " + p(
                percent_duplicate_timestamps) + " duplicate timestamps")
            print(f"DROPPED: {num_pisa_dropped} " + p(percent_pisa_dropped) + f" from {num_pisa_pre} " + p(
                percent_pisa_pre) + " pressure-induced sensor attenuation affected entries")
        else:
            print(f"\n\n")
            print(f"-----------------------------------------------------------------------")
            print(f"----------               DATA CLEANING REPORT                ----------")
            print(f"-----------------------------------------------------------------------")
            print(f"\n--------------------------    All Sensors    --------------------------")
            kpis_all = calculate_data_cleaning_kpis(self.files)
            output_table_all_sensors = generate_output_table(kpis_all, method='all_sensors')
            print(output_table_all_sensors)

            for sensor in self.sensors:
                print(f"\n---------------------------    " + sensor + "    ---------------------------")
                kpis_sensor = calculate_data_cleaning_kpis(self.files[self.files["sensor"] == sensor])
                output_table_one_sensor = generate_output_table(kpis_sensor, method='one_sensor', kpis_all=kpis_all)
                print(output_table_one_sensor)


def main():
    DataCleaner()


if __name__ == "__main__":
    main()
