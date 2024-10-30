import pandas as pd
import os
import glob as glob


class q56DataProcessing:

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
        file_list=glob.glob(self.raw_data_folder + '*.csv')
        
        # Put all dataframes into one
        self.scada_df = pd.DataFrame()
        for f in file_list:
            temp_df=pd.read_csv(f, low_memory=False)
            self.scada_df = pd.concat([self.scada_df, temp_df],
                                      ignore_index=True)
        # Process data
        self.process_data()
        
        # Save as csv
        filepath = os.path.join(scada_save_folder, "q56_scada.csv")
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
                                    f"q56_Turbine_{id_val}.csv")
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
        # Rename columns
        self.scada_df = self.scada_df.rename(
            columns={'TimeStamp': "datetime",
                     "Turbine ID": "turbine",
                     'Wind Speed Mean (m/s)': "wind_speed_ms",
                     "Active_Power_Mean": "power_kW",
                     "Yaw_Direction_Mean": "yaw_deg"})
        
        # Remove " from datetime column
        self.scada_df["datetime"] = self.scada_df["datetime"].astype(
            str).str.replace('"','')   
        # Convert turbine values to string, then correct erroneous
        # turbine values
        self.scada_df['turbine'] = self.scada_df['turbine'].astype(str)
        for i in self.scada_df['turbine'].unique():
            if i[-1] == ';':
                self.scada_df.loc[
                    self.scada_df['turbine'] == i,'turbine'] = '2'+i[:-1]
        
        # Convert power, and yaw field into float
        self.scada_df['power_kW'] = self.scada_df['power_kW'].astype(
            str).str.replace(',', '').astype(float)
        self.scada_df['yaw_deg'] = self.scada_df["yaw_deg"].astype(float)
        # Convert wind speed column to numeric, because of some Null string
        # values
        self.scada_df['wind_speed_ms'] = pd.to_numeric(self.scada_df[
            'wind_speed_ms'], errors='coerce')
        
        # Make datetime tz aware
        self.scada_df.set_index("datetime", inplace=True)
        self.scada_df.index = pd.to_datetime(self.scada_df.index,
                                             errors="coerce",
                                             format="%Y-%m-%d %H:%M:%S.%f")
        self.scada_df.index = self.scada_df.index.tz_localize(
            "America/Chicago", ambiguous="NaT", nonexistent="NaT")
        self.scada_df = self.scada_df.reset_index()

        # Remove duplicated timestamps and turbine id
        self.scada_df = self.scada_df.drop_duplicates(
            subset=["datetime", "turbine"], keep="first")
        self.scada_df = self.scada_df.sort_values(
            ["datetime", "turbine"]) # sort
        
        # Correct wind direction northing calibration
        northing_offsets = {
            "2301111": -29.995,
            "2301113": -36.07,
            "2301114": -16.65,
            "2301115": -25.708,
            "2301116": -14.932,
            "2301117": -85.084,
            "2301119": -25.674,
            "2301121": -2.705,
            "2301122": -17.187,
            "2301124": -21.748,
            "2301125": -26.773,
            "2301126": -20.212,
            "2301128": 11.285,
            "2301130": -14.011,
            "2301131": 13.965,
            "2301133": -25.222,
            "2301140": -14.72,
            "2301141": -31.413,
            "2301142": -38.046,
            "2301364": -31.475,
            "2301365": -27.014,
            "2301366": -31.213,
            "2301367": -20.026,
            "2301368": -22.11,
            "2301369": -20.936,
            "2301372": -33.105,
            "2301373": -26.261,
            "2301375": 7.753,
            "2301376": -0.253,
            "2301377": -0.339,
            "2301378": -17.213,
            "2301379": -14.022,
            "2301380": -23.468,
            "2301381": -8.399,
            "2301382": -12.31,
            "2301384": -7.276,
            "2301386": -10.896,
            "2301134": -9.187,
            "2301123": -56.268,
            "2301137": -10.161,
            "2301110": -8.555,
            "2301385": -4.423,
            "2301120": -8.661,
            "2301139": -22.08,
            "2301129": -7.301,
            "2301112": 19.118
        }
        
        for t in northing_offsets.keys():
            valid_inds = (self.scada_df["turbine"] == t)
            self.scada_df.loc[valid_inds,"yaw_deg"] = (
                self.scada_df.loc[valid_inds,"yaw_deg"] -
                northing_offsets[t]) % 360.0
