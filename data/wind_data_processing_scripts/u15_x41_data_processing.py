import pandas as pd
import os

class u15x41DataProcessing:
    
    def __init__(self, raw_data_directory):
        """
        self.raw_data_dir (str): Directory of either x41 (Panhandle 2) or
            u15 (st. Joseph) owner provided scada data.
        """
        self.raw_data_dir = raw_data_directory
        
    def scada_processing(self, scada_save_folder):
        """
        Processes the scada file for QA and plotting.
        Note: The scada data for x41 and u15 are both 8 GB.
        ----------
            scada_save_folder (str): Folder of where to save scada file.
        Return:
        ------
            None.
        """
        
        # Columns of interest
        columns = ["TimeStamp", "SCADA_WTG_Name",
                   'wtc_ActPower_mean', 'wtc_AcWindSp_mean',
                   'wtc_NacelPos_mean']
        # Large file so read and process the file in chunks
        df = pd.read_csv(self.raw_data_dir, usecols=columns,
                                    chunksize=1e6)
        temp_df = []
        for chunk in df:
            chunk = chunk.rename(columns={
                'TimeStamp': "datetime",
                "SCADA_WTG_Name": "turbine",
                "wtc_AcWindSp_mean": "wind_speed_ms",
                'wtc_ActPower_mean': "power_kW",
                'wtc_NacelPos_mean':"yaw_deg"})
            # Make datetime tz aware
            chunk.set_index("datetime", inplace=True)
            chunk.index = pd.to_datetime(chunk.index)
            chunk.index = chunk.index.tz_localize(
                "UTC", ambiguous="NaT", nonexistent="NaT")
            chunk.index = chunk.index.tz_convert("America/Chicago")
            chunk = chunk.reset_index()

            self.process_data(chunk)
            
            temp_df.append(chunk)
            #print('one chunk done')
            
        self.scada_df = pd.concat(temp_df, ignore_index=True)
        
        # Save as csv
        if "pan" in self.raw_data_dir.lower():
            filepath = os.path.join(scada_save_folder, "x41_scada.csv")
        if "sjw" in self.raw_data_dir.lower():
            filepath = os.path.join(scada_save_folder, "u15_scada.csv")
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
            if "pan" in self.raw_data_dir.lower():
                filepath = os.path.join(turbine_save_folder,
                                        f"x41_Turbine_{id_val}.csv")
            if "sjw" in self.raw_data_dir.lower():
                filepath = os.path.join(turbine_save_folder,
                                        f"u15_Turbine_{id_val}.csv")

            group_df.to_csv(filepath, index=False)
    
    def process_data(self, df):
        """
        Processes and standardizes the scada data.
        Converts columns to their appropriate dtypes, removes any duplications,
        and corrects yaw offset due to incorrect northing calibration.
        Parameter:
        ----------
            df (dataframe): Pandas dataframe with power_kW, datetime, and
                turbine columns.
        Return:
        ------
            df (dataframe): Pandas dataframe after data processing.
        """
        # Convert turbine column to string
        df["turbine"] = df["turbine"].astype(str)
        # Convert power, wind speed, and yaw field into float
        df['power_kW'] = df['power_kW'].astype(
            str).str.replace(',', '').astype(float)
        df['wind_speed_ms'] = df['wind_speed_ms'].astype(float)
        df['yaw_deg'] = df["yaw_deg"].astype(float)
        
        # Correct offsets in yaw_deg
        if "sjw" in self.raw_data_dir.lower():
            northing_offsets = {
                "SJ-007": 5.274,
                "SJ-008": 9.252,
                "SJ-009": 8.03,
                "SJ-011": -0.917,
                "SJ-012": -0.413,
                "SJ-019": -3.16,
                "SJ-020": -2.794,
                "SJ-021": -3.532,
                "SJ-023": -6.699,
                "SJ-024": -5.885,
                "SJ-025": 16.31,
                "SJ-030": -3.117,
                "SJ-031": -4.0,
                "SJ-032": 4.055,
                "SJ-035": 0.272,
                "SJ-036": -4.174,
                "SJ-037": 1.012,
                "SJ-038": -2.061,
                "SJ-041": 3.469,
                "SJ-042": -2.291,
                "SJ-043": 4.289,
                "SJ-044": -1.475,
                "SJ-045": 1.26,
                "SJ-047": 8.567,
                "SJ-050": 11.238,
                "SJ-051": 3.042,
                "SJ-052": 0.679,
                "SJ-053": 1.863,
                "SJ-054": 7.891,
                "SJ-058": -2.711,
                "SJ-059": -3.553,
                "SJ-060": 3.294,
                "SJ-061": -1.014,
                "SJ-062": 4.7,
                "SJ-063": 5.916,
                "SJ-064": 3.669,
                "SJ-065": 1.141,
                "SJ-068": -0.761,
                "SJ-069": -2.665,
                "SJ-070": 2.167,
                "SJ-071": 21.687,
                "SJ-073": 9.212,
                "SJ-074": -3.139,
                "SJ-075": 12.822,
                "SJ-076": 1.209,
                "SJ-077": -6.421,
                "SJ-078": -1.146,
                "SJ-079": 5.284,
                "SJ-080": 2.749
            }
        if "pan" in self.raw_data_dir.lower():
            northing_offsets = {
                "P202": -0.269,
                "P205": -0.167,
                "P208": -12.509,
                "P209": -11.664,
                "P210": 7.215,
                "P211": -17.041,
                "P212": -5.889,
                "P215": -37.845,
                "P216": -9.258,
                "P220": -15.581,
                "P222": -9.501,
                "P223": -8.594,
                "P224": -7.156,
                "P225": 23.48,
                "P228": 1.072,
                "P231": -11.555,
                "P234": -11.339,
                "P235": -10.777,
                "P237": -12.044,
                "P239": -16.708,
                "P242": -7.224,
                "P243": -37.305,
                "P244": -22.733,
                "P248": -9.937,
                "P251": -3.203,
                "P252": -8.361,
                "P253": 40.352,
                "P256": -17.41,
                "P259": -5.981,
                "P263": -0.392,
                "P269": -10.111,
                "P273": -10.989,
                "P276": -6.459,
                "P277": 0.089,
                "P278": -20.095,
                "P279": -14.016,
                "P280": -4.027,
            }
        for t in northing_offsets.keys():
            valid_inds = (df["turbine"] == t)
            df.loc[valid_inds,"yaw_deg"] = (df.loc[
                valid_inds,"yaw_deg"] - northing_offsets[t]) % 360.0
            
        # Remove any and all duplications in the dates for a given turbine
        df.drop_duplicates(
            subset=['datetime', 'turbine'], inplace=True, keep="first")
            
        return df
        
