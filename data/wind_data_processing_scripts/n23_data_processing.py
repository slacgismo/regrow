import pandas as pd
import os
import numpy as np


class n23DataProcessing:

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

        # Read and join consecutive years for wind speed, wind direction and power
        # All wind speed data
        df_arr = [pd.read_csv(
            (self.raw_data_folder +
             '/WTG Wind Speed/ORIGIN WTG WIND SPEED DATA 201808 YTD.TXT'),
            sep=';', engine='python')]
        for i in ['2015', '2016', '2017']:
            wind_speed = pd.read_csv(
                (self.raw_data_folder +
                 '/WTG Wind Speed/ORIGIN WTG WIND SPEED DATA ' + i + '.TXT'),
                sep=';', engine='python')
            df_arr.append(wind_speed)
        self.ws = pd.concat(df_arr, ignore_index=True)
        # All wind direction data
        df_arr = [pd.read_csv(
            (self.raw_data_folder +
             '/WTG Wind Direction/ORIGIN WTG WIND DIR DATA 201808 YTD.TXT'),
            sep=';', engine='python')]
        for i in ['2015', '2016', '2017']:
            wind_dir = pd.read_csv(
                (self.raw_data_folder +
                 '/WTG Wind Direction/ORIGIN WTG WIND DIR DATA ' + i + '.TXT'),
                sep=';', engine='python')
            df_arr.append(wind_dir)
        self.wd = pd.concat(df_arr, ignore_index=True)

        # All power data
        df_arr = [pd.read_csv(
            (self.raw_data_folder +
             '/WTG Power/ORIGIN WTG POWER DATA 201808 YTD.TXT'),
            sep=';', engine='python')]
        for i in ['2015', '2016', '2017']:
            power = pd.read_csv(
                (self.raw_data_folder +
                 '/WTG Power/ORIGIN WTG POWER DATA ' + i + '.TXT'),
                sep=';', engine='python')
            df_arr.append(power)
        self.w = pd.concat(df_arr, ignore_index=True)
        
        # Process data
        self.process_data()
        
        # Save as csv
        filepath = os.path.join(scada_save_folder, "n23_scada.csv")
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
                                    f"n23_Turbine_{id_val}.csv")
            group_df.to_csv(filepath, index=False)
        
    def process_data(self):
        """
        Processes and standardizes the scada data.
        Standardizes column names, makes datetime timezone aware,
        converts columns to their appropriate dtypes, removes any duplications
        and corrects wind_dir_deg.
        Parameter:
        ----------
            None.
        Return:
        ------
            None.
        """
        # Drop irrelavant columns
        self.ws.drop(axis=1, columns=['Unnamed: 76'], inplace=True)
        self.wd.drop(axis=1, columns=['Unnamed: 76'], inplace=True)
        self.w.drop(axis=1, columns=['Unnamed: 76'], inplace=True)

        # Convert dataframe from wide to long format to make 2 columns only: 
        # Unit (turbine name) column and data (power, wind speed, wind dir) column
        ws_unit = self.ws.melt(id_vars=["PCTimeStamp       "],
                          var_name='Unit', value_name='AmbientWindSpeedAvg.')

        wd_unit = self.wd.melt(id_vars=["PCTimeStamp       "],
                          var_name='Unit', value_name='AmbientWindDirAbsoluteAvg.')
        w_unit = self.w.melt(id_vars=["PCTimeStamp       "],
                        var_name='Unit', value_name='GridProductionPowerAvg.')
        
        # Prepare Unit Column to be converted to a float
        wd_unit['Unit']= wd_unit['Unit'].apply(lambda x: x[1:3])
        ws_unit['Unit']= ws_unit['Unit'].apply(lambda x: x[1:3])
        w_unit['Unit']= w_unit['Unit'].apply(lambda x: x[1:3])
        # merge three variables into single dataframe
        df = wd_unit.merge(ws_unit, on=["Unit", 'PCTimeStamp       '])
        self.scada_df = df.merge(w_unit, on=["Unit", 'PCTimeStamp       '])
        
        # Rename columns to standardized names
        self.scada_df = self.scada_df.rename(columns={
            'PCTimeStamp       ': "datetime",
            "Unit": "turbine",
            "AmbientWindSpeedAvg.": "wind_speed_ms",
            'GridProductionPowerAvg.': "power_kW",
            'AmbientWindDirAbsoluteAvg.':"wind_dir_deg"})
        
        # Make datetime tz aware 
        self.scada_df['datetime'] = self.scada_df[
            'datetime'].astype(str).str.replace(' ', '')
        # Add midnight timestamp for new days, which does not automatically
        # have timestamps
        self.scada_df['datetime'] = self.scada_df['datetime'].apply(
            lambda x: self.stringEdit(x))
        self.scada_df.set_index("datetime", inplace=True)
        self.scada_df.index = pd.to_datetime(self.scada_df.index,
                                             errors="coerce",
                                             format = '%m/%d/%Y%I:%M:%S%p')
        self.scada_df.index = self.scada_df.index.tz_localize(
            "America/Chicago", ambiguous="NaT", nonexistent="NaT")
        self.scada_df = self.scada_df.reset_index()
        
        #convert datatypes
        self.scada_df['turbine'] = self.scada_df['turbine'].astype(
            int).astype(str)
        for c in ["power_kW", 'wind_dir_deg',
                  'wind_speed_ms']:
            self.scada_df[c] = self.scada_df[c].astype(str).str.replace(' ', '')
            self.scada_df[c]= self.scada_df[c].replace('', np.nan)
            self.scada_df[c]= self.scada_df[c].astype(float)

        # Correct wind direction northing calibration
        northing_offsets = {
            "1": -166.611,
            "2": -136.886,
            "3": -158.726,
            "4": -153.299,
            "5": -116.934,
            "6": -150.827,
            "7": -148.373,
            "8": -160.866,
            "9": -146.768,
            "10": -150.549,
            "11": -168.677,
            "12": -132.192,
            "14": -146.34,
            "15": -139.41,
            "16": -166.693,
            "17": -146.454,
            "18": -152.384,
            "19": -147.28,
            "21": -146.155,
            "23": -155.452,
            "24": 172.501,
            "26": -146.203,
            "27": -166.295,
            "28": 175.894,
            "29": -151.066,
            "31": -146.789,
            "32": 174.433,
            "33": -148.514,
            "35": -147.6,
            "38": -151.914,
            "39": 174.569,
            "40": -175.079,
            "41": -129.779,
            "43": -148.882,
            "44": 175.398,
            "45": -158.384,
            "49": 177.381,
            "50": 153.287,
            "51": -86.545,
            "52": -178.122,
            "53": 174.947,
            "54": -147.267,
            "55": -177.248,
            "57": -168.864,
            "58": -165.159,
            "60": -168.772,
            "61": -135.953,
            "62": -169.936,
            "63": -166.051,
            "65": 172.897,
            "66": 166.074,
            "68": 179.642,
            "69": 160.018,
            "70": -154.28,
            "71": -77.89,
            "72": -87.749,
            "73": -142.648,
            "74": -141.518,
            "75": -139.52
            }

        for t in northing_offsets.keys():
            valid_inds = (self.scada_df["turbine"] == t)
            self.scada_df.loc[valid_inds,"wind_dir_deg"] = (
                self.scada_df.loc[valid_inds,"wind_dir_deg"] -
                northing_offsets[t]) % 360.0
                
    def stringEdit(self, string):
        """
        If last character of string does not have an M at the end, then add
        12:00:00AM. This is for datetime that only has the date and
        does not have a midnight timestamp.
        """
        if string[-1] != 'M':
            string += '12:00:00AM'
        return string
