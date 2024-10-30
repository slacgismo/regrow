import pandas as pd
import os


class c43DataProcessing:

    def __init__(self, raw_data_directory):
        """
        self.raw_data_dir (str): Directory of c43 (Spinning Spur 1)
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
        self.scada_df = pd.read_csv(self.raw_data_dir)
        self.scada_df = self.scada_df[['timestamp', "WTG", 'wtc_ActPower_mean',
                                       'wtc_AcWindSp_mean', 'wtc_NacelPos_mean'
                                       ]]

        self.process_data()

        # Save as csv
        filepath = os.path.join(scada_save_folder, "c43_scada.csv")
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
                                    f"c43_Turbine_{id_val}.csv")
            group_df.to_csv(filepath, index=False)

    def process_data(self):
        """
        Processes and standardizes the scada data.
        Standardizes column names, makes datetime timezone aware,
        converts columns to their appropriate dtypes, removes duplications,
        and corrects yaw_deg due to incorrect northing calibrations.
        Parameter:
        ----------
            None.
        Return:
        ------
            None.
        """
        self.scada_df = self.scada_df.rename(
            columns={'timestamp': "datetime",
                     "WTG": "turbine",
                     'wtc_AcWindSp_mean': "wind_speed_ms",
                     'wtc_ActPower_mean': "power_kW",
                     "wtc_NacelPos_mean": "yaw_deg"})
        
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

        # Correct wind direction northing calibration
        northing_offsets = {
            "5": -24.356,
            "10": -28.463,
            "17": -7.144,
            "24": -16.899,
            "36": -39.416,
            "43": -25.638,
            "55": -33.873,
            "62": -19.149,
            "6": 5.002,
            "13": -11.914,
            "18": -12.671,
            "25": -16.089,
            "32": -6.723,
            "37": -19.524,
            "44": -28.54,
            "51": -0.658,
            "63": -35.784,
            "70": -39.868,
            "3": -4.666,
            "8": -6.374,
            "15": 13.463,
            "22": -5.741,
            "34": -109.594,
            "41": -25.415,
            "53": -7.287,
            "60": -15.417,
            "45": -27.277,
            "50": -21.056,
            "57": -12.455,
            "64": -31.523,
            "69": -0.786,
            "2": -15.279,
            "9": 1.47,
            "21": -17.685,
            "28": -26.228,
            "40": -15.052,
            "47": -22.657,
            "54": -8.957,
            "59": -23.907,
            "66": -8.564,
            "1": -21.666,
            "20": -9.539,
            "27": -12.021,
            "39": -28.396,
            "46": -11.023,
            "58": -25.0,
            "65": -29.244,
            "12": -16.529,
            "19": -10.0,
            "31": -0.207,
            "38": -38.147,
            "14": -65.412,
            "26": -13.787,
            "11": 3.353,
            "30": -28.961,
            "49": -0.771,
            "56": -20.31,
            "68": -16.508,
            "4": -79.175,
            "16": -13.984,
            "23": -37.853,
            "42": -17.073,
            "61": -14.644,
            "33": -126.855,
            "52": -25.85,
            "29": -21.949,
            "48": -13.228,
            "67": -1.278
        }

        for t in northing_offsets.keys():
            valid_inds = (self.scada_df["turbine"] == t)
            self.scada_df.loc[valid_inds, "yaw_deg"] = (
                self.scada_df.loc[valid_inds, "yaw_deg"] - northing_offsets[t]
            ) % 360.0
        
        # Remove any and all duplications in the dates for a given turbine
        self.scada_df.drop_duplicates(
            subset=['datetime', 'turbine'], inplace=True, keep="first")
