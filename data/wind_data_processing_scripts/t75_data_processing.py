import pandas as pd
import os


class t75DataProcessing:

    def __init__(self, raw_data_directory):
        """
        self.raw_data_dir (str): Directory of t75 (Meridian Way)
            owner provided scada data.
        """
        self.raw_data_dir = raw_data_directory

    def scada_processing(self, scada_save_folder):
        """
        Processes the scada file for QA and plotting.
        ----------
            scada_save_folder (str): Folder of where to save scada file.
        Return:
        ------
            None.
        """
        self.scada_df = pd.read_csv(self.raw_data_dir,
                                    skiprows=1,
                                    names=['DateTime', 'TurbineID',
                                           'Velocity_ms', 'Yaw_deg',
                                           'Power_kW'])

        self.process_data()

        # Save as csv
        filepath = os.path.join(scada_save_folder, "t75_scada.csv")
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
            # Save as csv
            filepath = os.path.join(turbine_save_folder,
                                    f"t75_Turbine_{id_val}.csv")
            group_df.to_csv(filepath, index=False)

    def process_data(self):
        """
        Processes and standardizes the scada data.
        Standardizes column names, makes datetime timezone aware,
        converts columns to their appropriate dtypes, and removes duplications.
        Parameter:
        ----------
            None.
        Return:
        ------
            None.
        """
        self.scada_df = self.scada_df.rename(
            columns={'DateTime': "datetime",
                     "TurbineID": "turbine",
                     'Velocity_ms': "wind_speed_ms",
                     'Power_kW': "power_kW",
                     "Yaw_deg": "yaw_deg"})
        
        # Remove DST time offsets.
        # Note in 2016 this for some reason happens at European dates.
        # The rest occur on US dates.
        self.scada_df['datetime'] = pd.to_datetime(self.scada_df['datetime'],
                                                   errors="coerce")
        self.scada_df.loc[
            (self.scada_df["datetime"] >= pd.to_datetime("2014-03-09 03:00")) &
            (self.scada_df["datetime"] < pd.to_datetime("2014-11-02 01:00")),
            "datetime"]-= pd.Timedelta(hours = 1)
        self.scada_df.loc[
            (self.scada_df["datetime"] >= pd.to_datetime("2015-03-08 03:00")) &
            (self.scada_df["datetime"] < pd.to_datetime("2015-11-01 01:00")),
            "datetime"] -= pd.Timedelta(hours = 1)
        self.scada_df.loc[
            (self.scada_df["datetime"] >= pd.to_datetime("2016-03-26 20:00")) &
            (self.scada_df["datetime"] < pd.to_datetime("2016-10-29 18:00")),
            "datetime"] -= pd.Timedelta(hours = 1)
        self.scada_df.loc[
            (self.scada_df["datetime"] >= pd.to_datetime("2017-03-12 03:00")),
            "datetime"] -= pd.Timedelta(hours = 1)
        
        # Check if there are duplicated timestamps, and if so, drop the
        # duplicates for each turbine
        self.scada_df = self.scada_df.drop_duplicates(
            subset=["datetime", "turbine"], keep="last")
        self.scada_df = self.scada_df.sort_values(["datetime", "turbine"])
        
        # remove undefined times
        self.scada_df.dropna(subset=['datetime'], inplace=True)
        
        # Make datetime tz aware
        self.scada_df.set_index("datetime", inplace=True)
        self.scada_df.index = pd.to_datetime(self.scada_df.index)
        self.scada_df.index = self.scada_df.index.tz_localize(
            "America/Chicago", ambiguous="NaT", nonexistent="NaT")
        self.scada_df = self.scada_df.reset_index()
        
        # Convert turbine column to string
        self.scada_df["turbine"] = self.scada_df["turbine"].astype(str)
        # Convert power, wind, and yaw field into float
        self.scada_df['power_kW'] = self.scada_df['power_kW'].astype(
            str).str.replace(',', '').astype(float)
        self.scada_df['wind_speed_ms'] = self.scada_df[
            'wind_speed_ms'].astype(float)
        self.scada_df['yaw_deg'] = self.scada_df["yaw_deg"].astype(float)
