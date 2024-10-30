import os
import pandas as pd
import glob as glob


class w34DataProcessing:

    def __init__(self, raw_data_folder):
        """
        self.raw_data_folder (str): Folder of raw power, wind speed, wind
            direction, and yaw data.
        """
        self.raw_data_folder = raw_data_folder

    def scada_processing(self, scada_save_folder):
        """
        Combines the various SCADA csv files into one SCADA csv file and
        process that file for QA and plotting.
        Parameter:
        ----------
            scada_save_folder (str): Folder of where to save scada file.
        Return:
        ------
            None.
        """
        # Get all csv scada files
        file_list = glob.glob(self.raw_data_folder + '*.csv')

        # Put all dataframes into one
        self.scada_df = pd.DataFrame()
        for f in file_list:
            temp_df = pd.read_csv(f, sep=";", low_memory=False)
            self.scada_df = pd.concat([self.scada_df, temp_df],
                                      ignore_index=True)

        # Process data
        self.process_data()

        # Save as csv
        filepath = os.path.join(scada_save_folder, "w34_scada.csv")
        self.scada_df.to_csv(filepath, index=False)

    def generate_turbine_lvl_files(self, turbine_save_folder):
        """
        Reformats the scada data by turbine level.
        Parameter:
        ----------
            turbine_save_folder (str): Folder of where to save turbine
                level files.
        Return:
        ------
            None.
        """
        # Get all row values for an id
        tur_df = self.scada_df.groupby("turbine")
        for id_val, group_df in tur_df:
            filepath = os.path.join(turbine_save_folder,
                                    f"w34_Turbine_{id_val}.csv")
            group_df.to_csv(filepath, index=False)

    def process_data(self):
        """
        Processes and standardizes the scada data.
        Standardizes column names, makes datetime timezone aware,
        converts columns to their appropriate dtypes, removes any duplications,
        and corrects yaw_deg due to incorrect northing calibrations.
        Parameter:
        ----------
            None.
        Return:
        ------
            None.
        """
        # Get only columns of interest
        self.scada_df = self.scada_df[['TimeStampUTCSystem', 'systemnumber',
                                       'Power', 'Wind speed',
                                       'Nacelle position']]
        # Rename columns
        self.scada_df = self.scada_df.rename(
            columns={'TimeStampUTCSystem': "datetime",
                     "systemnumber": "turbine",
                     'Wind speed': "wind_speed_ms",
                     "Power": "power_kW",
                     "Nacelle position": "yaw_deg"})

        # Make datetime tz aware
        self.scada_df.set_index("datetime", inplace=True)
        self.scada_df.index = pd.to_datetime(self.scada_df.index,
                                             errors="coerce",
                                             format="%Y-%m-%d %H:%M:%S.%f")
        self.scada_df.index = self.scada_df.index.tz_localize(
            "America/Chicago", ambiguous="NaT", nonexistent="NaT")
        self.scada_df = self.scada_df.reset_index()
        # remove undefined times after timezone conversion
        self.scada_df.dropna(subset=['datetime'], inplace=True)

        # convert columns to correct format
        cols = ['wind_speed_ms', 'yaw_deg', 'power_kW']
        for c in cols:
            self.scada_df[c] = self.scada_df[c].astype(
                str).str.replace(',', '.').astype(float)

        # Remove turbine ids that don't seem to correspond to turbines in
        # the wind farm
        self.scada_df['turbine'] = self.scada_df['turbine'].astype(
            int)
        self.scada_df = self.scada_df[
            ~self.scada_df["turbine"].isin([1003185210, 1003185240])]
        # Change turbine column to string format
        self.scada_df['turbine'] = self.scada_df['turbine'].astype(
            str)

        # Remove duplicated timestamps and turbine id
        self.scada_df = self.scada_df[
            self.scada_df.duplicated(subset=['datetime', 'turbine']) == False]

        # Remove times that are between regular 10-minute sampling intervals
        self.scada_df = self.scada_df[
            ((self.scada_df["datetime"].dt.minute % 10) == 0) &
            (self.scada_df["datetime"].dt.second == 0)]

        # Correct northing calibration of wind directions based on estimated
        # offsets for valid time range
        start_time = "2014-01-01 00:00"
        end_time = "2016-11-28 12:00"

        northing_offsets = {
            "16111436": 22.543,
            "16111439": -4.063,
            "16111440": 8.324,
            "16111441": -0.921,
            "16111442": 3.044,
            "16111443": 41.886,
            "16111444": -5.034,
            "16111445": -3.183,
            "16111446": -2.691,
            "16111447": -0.473,
            "16111449": 4.265,
            "16111450": -25.135,
            "16111451": 4.043,
            "16111452": 9.099,
            "16111454": -3.69,
            "16111455": 74.697,
            "16111456": 164.132,
            "16111457": 3.334,
            "16111458": 0.0,
            "16111459": 2.603,
            "16111460": -3.277,
            "16111461": 1.766,
            "16111462": -24.785,
            "16111463": -10.029,
            "16111465": 4.036,
            "16111466": -36.754,
            "16111467": 88.504,
            "16111468": -6.104,
            "16111469": 10.624,
            "16111470": 3.242,
            "16111471": 3.049,
            "16111472": 4.994,
            "16111474": -13.875,
            "16111476": -8.305,
            "16111477": -9.249,
            "16111478": 4.391,
            "16111479": 6.081,
            "16111480": -3.498,
            "16111481": -33.36,
            "16111482": 4.278,
            "16111483": -1.963,
            "16111484": 12.69,
            "16111486": 6.73,
            "16111487": -4.078,
            "16111488": 23.504,
            "16111489": -2.476,
            "16111490": -10.839,
            "16111491": 9.477,
            "16111492": -3.504,
            "16111493": 3.476,
            "16111494": -0.153,
            "16111497": 14.003,
            "16111498": 26.396,
            "16111501": -3.636,
            "16111502": -0.681,
            "16111503": -2.049,
            "16111504": 0.504,
            "16111505": 5.438,
            "16111506": -5.99,
            "16111507": 7.905,
            "16111508": 11.816,
            "16111509": -11.807,
            "16111510": 8.853,
            "16111511": -3.909,
            "16111512": -10.724,
            "16111513": 3.356,
            "16111514": -9.373,
            "16111515": 1.526,
            "16111516": 0.525,
            "16111517": 6.172,
            "16111518": -7.787,
            "16111519": 5.806,
            "16111522": -80.973,
            "16111524": -8.115,
            "16111525": 4.878,
            "16111527": 14.391,
            "16111528": 4.17,
            "16111529": -1.328
        }
        for t in northing_offsets.keys():
            valid_inds = ((self.scada_df["turbine"] == t) &
                          (self.scada_df["datetime"] >= start_time) &
                          (self.scada_df["datetime"] < end_time))
            self.scada_df.loc[valid_inds, "yaw_deg"] = (self.scada_df.loc[
                valid_inds, "yaw_deg"] - northing_offsets[t]) % 360.0
