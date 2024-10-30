import pandas as pd
import os


class r47DataProcessing:

    def __init__(self, raw_data_folder):
        """
        self.raw_data_folder (str): Folder of raw csv and txt data.
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
        # SCADA data file 1
        df_scada1 = pd.read_csv(
            self.raw_data_folder +
            'Prairie-Star_turbine-data_2012.txt',
            skiprows=0, sep=';')
        # convert European style of decimal separator (,) to
        # normal decimal separate (.)
        df_scada1['yaw_deg'] = df_scada1['Yaw'].astype(
            str).apply(lambda x: x.replace(',', '.'))
        df_scada1['power_kW'] = df_scada1['ActivePower'].astype(
            str).apply(lambda x: x.replace(',', '.'))
        df_scada1['wind_speed_ms'] = df_scada1['WindSpeed'].astype(
            str).apply(lambda x: x.replace(',', '.'))
        # Rename columns
        df_scada1 = df_scada1.rename(
            columns={'DATE': 'datetime',
                     'WT': 'turbine'})
        # Keep columns of interest
        df_scada1 = df_scada1.drop(
            ['WF', 'Status_Max', 'Status_Min', 'Yaw', 'ActivePower',
             'WindSpeed'], axis=1)

        # SCADA data file 2
        df_scada2 = pd.read_csv(
            self.raw_data_folder +
            'Prairie-Star_turbine-data.csv', skiprows=0, sep=',')
        # Rename columns
        df_scada2 = df_scada2.rename(
            columns={'Fecha_hora': 'datetime',
                     'turbina': 'turbine',
                     'Vel': 'wind_speed_ms',
                     'YAW': 'yaw_deg',
                     'Pot_real': 'power_kW'})

        # combine both
        self.scada_df = pd.concat([df_scada1, df_scada2])[
            df_scada2.columns.tolist()]
        
        # process data
        self.process_data()
        
        # Save as csv
        filepath = os.path.join(scada_save_folder, "r47_scada.csv")
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
                                    f"r47_Turbine_{id_val}.csv")
            group_df.to_csv(filepath, index=False)

    def process_data(self):
        """
        Processes and standardizes the scada data.
        Makes datetime timezone aware, converts columns to their
        appropriate dtypes, and removes duplications.
        Parameter:
        ----------
            None.
        Return:
        ------
            None.
        """

        # Remove duplicate timestamps
        self.scada_df = self.scada_df[
            ~(((self.scada_df['datetime'] > '2011 10-29 19:00:00') &
               (self.scada_df['datetime'] < '2011 10-29 20:10:00')) |
              ((self.scada_df['datetime'] > '2012 10-27 19:00:00') &
               (self.scada_df['datetime'] < '2012 10-27 20:10:00')))]

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
        
        # Remove any and all duplications in the dates for a given turbine
        self.scada_df.drop_duplicates(
            subset=['datetime', 'turbine'], inplace=True, keep="first")
        self.scada_df = self.scada_df.sort_values(["datetime", "turbine"])
