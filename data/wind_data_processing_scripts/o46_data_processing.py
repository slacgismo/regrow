import pandas as pd
import os


class o46DataProcessing:

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
        # Maps column names to file containing the data
        field_dict = {
            'power_kW': 'nrel data - turb - power.csv',
            'wind_speed_ms': 'nrel data - turb - wdspd.csv',
            'wind_dir_deg': 'nrel data - turb - wddir (calc).csv',
            'yaw_deg': 'nrel data - turb - yaw_corrected (calc).csv',
        }

        self.scada_df = pd.DataFrame()

        # Get and read all csv files from field dict
        for field, filename in field_dict.items():
            filepath = os.path.join(self.raw_data_folder, filename)
            if (field == 'wind_dir_deg') | (field == 'yaw_deg'):
                # Get data
                df = pd.read_csv(filepath, header=None,
                                 skiprows=2, low_memory=False)
                # Get header names
                df_names = pd.read_csv(
                    filepath, header=None, skiprows=1,
                    nrows=1, low_memory=False)
                df_names_sub = df_names.squeeze()

            else:
                # Get data
                df = pd.read_csv(filepath, header=None,
                                 skiprows=6, low_memory=False)
                # Get header names
                df_names = pd.read_csv(filepath, header=None,
                                       skiprows=4, nrows=1, low_memory=False)
                df_names_sub = df_names.squeeze()
                # Get only turbine names from tag
                df_names_sub = df_names_sub.str.slice(4, 9)

            df_names_sub[0] = 'datetime'

            df.columns = df_names_sub
            # Make datetime tz aware
            df.set_index("datetime", inplace=True)
            df.index = pd.to_datetime(df.index, format='%m/%d/%Y %H:%M')
            df.index = df.index.tz_localize(
                "America/Chicago", ambiguous="NaT", nonexistent="NaT")

            temp_df = pd.DataFrame(columns=["datetime", 'turbine'])

            for turbine in df_names_sub[1:]:
                turbine_df = pd.DataFrame(
                    data={'datetime': df.index,
                          'turbine': turbine,
                          field: df[turbine]
                          })
                turbine_df = turbine_df.dropna()
                temp_df = pd.concat([temp_df, turbine_df], ignore_index=True)

            # If first field object, create new scada data frame
            # If not, merge with scada df
            if field == list(field_dict.keys())[0]:
                self.scada_df = temp_df
            else:
                self.scada_df = self.scada_df.merge(
                    temp_df, on=["datetime", 'turbine'])

        # Process data
        self.process_data()

        filepath = os.path.join(scada_save_folder, "o46_scada.csv")
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
                                    f"o46_Turbine_{id_val}.csv")
            group_df.to_csv(filepath, index=False)

    def process_data(self):
        """
        Processes and standardizes the scada data.
        Converts columns to their appropriate dtypes, removes any duplications
        and corrects wind_dir_deg.
        Parameter:
        ----------
            None.
        Return:
        ------
            None.
        """
        # Convert turbine column to string
        self.scada_df["turbine"] = self.scada_df["turbine"].astype(str)
        # Convert power, wind speed, wind direction, and yaw field into float
        self.scada_df['power_kW'] = self.scada_df['power_kW'].astype(
            str).str.replace(',', '').astype(float)
        self.scada_df['wind_speed_ms'] = self.scada_df[
            'wind_speed_ms'].astype(float)
        self.scada_df['yaw_deg'] = self.scada_df["yaw_deg"].astype(float)
        self.scada_df['wind_dir_deg'] = self.scada_df[
            "wind_dir_deg"].astype(float)
        
        # Correct wind direction northing calibration
        self.scada_df["wind_dir_deg"] = (
            self.scada_df["wind_dir_deg"] - -2.078) % 360.0
        
        # Remove any and all duplications in the dates for a given turbine
        self.scada_df.drop_duplicates(
            subset=['datetime', 'turbine'], inplace=True, keep="first")
