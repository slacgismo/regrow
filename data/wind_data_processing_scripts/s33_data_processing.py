import pandas as pd
import os


class s33DataProcessing:

    def __init__(self, raw_data_directory):
        """
        self.raw_data_dir (str): Directory of s33 (Los Mirasoles)
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
        self.scada_df = pd.read_csv(self.raw_data_dir, sep = ";")

        self.process_data()

        # Save as csv
        filepath = os.path.join(scada_save_folder, "s33_scada.csv")
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
                                    f"s33_Turbine_{id_val}.csv")
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
            columns={'Fecha_hora': "datetime",
                     "turbina": "turbine",
                     'Vel': "wind_speed_ms",
                     "Pot_real": "power_kW",
                     "YAW": "yaw_deg"})
        
        # Make datetime tz aware
        self.scada_df.set_index("datetime", inplace=True)
        self.scada_df.index = pd.to_datetime(self.scada_df.index,
                                             errors="coerce")
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
        
        # Check if there are duplicated timestamps, and if so, drop the
        # duplicates for each turbine
        self.scada_df = self.scada_df.drop_duplicates(
            subset=["datetime", "turbine"], keep="last")
        self.scada_df = self.scada_df.sort_values(["datetime", "turbine"])
