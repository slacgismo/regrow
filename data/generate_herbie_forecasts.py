import pandas as pd
#from pvlib.forecast import HRRR
from herbie import Herbie

from herbie.toolbox import EasyMap, pc
from herbie import paint
import numpy as np
import matplotlib.pyplot as plt
import shutil
from dask import delayed
import os

@delayed
def pull_herbie_hrr_data(date, time_horizon):
    """
    Function for pulling Herbie data, which we will be parallelizing via Dask.
    """
    H = Herbie(date,
                model="hrrr",
                product="prs",
                fxx=time_horizon
            )
    H.download()
    # ":TCDC:entire atmosphere:anl": overall cloud cover
    #:UGRD:80 m above ground:anl: u-component wind speed (80 m above ground)
    #:VGRD:80 m above ground:anl: v-component wind speed (80 m above ground)
    #":RH:2 m above ground:anl": relative humidity at surface
    #TMP:surface:anl: surface temperature
    #:PRES:surface:anl: surface pressure
    #DPT:2 m above ground:anl: dew point 
    # Full list of options: https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/HRRR_archive/hrrr_sfc_table_f00-f01.html
    tags = [":TCDC:entire atmosphere:" + str(time_horizon) + " hour fcst",
            ":UGRD:80 m above ground:" + str(time_horizon) + " hour fcst",
            ":VGRD:80 m above ground:" + str(time_horizon) + " hour fcst",
            ":RH:2 m above ground:" + str(time_horizon) + " hour fcst",
            ":PRES:surface:" + str(time_horizon) + " hour fcst",
            ":TMP:surface:" + str(time_horizon) + " hour fcst",
            ":DPT:2 m above ground:" + str(time_horizon) + " hour fcst"]
    for tag in tags:
        ds = H.xarray(tag,remove_grib=True)
        # get all of the closest forecasts to the WECC node points
        dsi = ds.herbie.nearest_points(points=points,
                                        names=names)
        # Build out a dataframe for the predictions
        pred_df = pd.DataFrame()
        pred_df['county'] = list(dsi.point.values)
        pred_df['point_latitude'] = list(dsi.point_latitude.values)
        pred_df['point_longitude'] = list(dsi.point_longitude.values)
        pred_df['tag'] = tag
        pred_df['forecast_time'] = date
        pred_df['forecast_horizon_hrs'] = time_horizon
        if "d2m" in dsi:
            pred_df['value'] = list(dsi.d2m.values)
        elif "tcc" in dsi:
            pred_df['value'] = list(dsi.tcc.values)
        elif "u" in dsi:
            pred_df['value'] = list(dsi.u.values)
        elif "v" in dsi:
            pred_df['value'] = list(dsi.v.values)
        elif "r2" in dsi:
            pred_df['value'] = list(dsi.r2.values)
        elif "sp" in dsi:
            pred_df['value'] = list(dsi.sp.values)
        elif "t" in dsi:
            pred_df['value'] = list(dsi.t.values)
    return pred_df

@delayed
def pull_herbie_gefs_data(data, time_horizon):
    """
    Similar function for Herbie GEFS data.
    """
    H = Herbie(date,
               model="gefs",
               fxx=time_horizon,
               product="atmos.5b",
               member="p01",
            )
    H.download()
    tags = ["UGRD:80 m",
            "VGRD:80 m",
            "TCDC",
            #"RH",
            #"PRES",
            "TMP:surface",
            "DPT:2 m"]
    specific_tag_names = [":TCDC:475 mb:",
                            ":UGRD:80 m above ground:",
                            ":VGRD:80 m above ground:",
                            #":RH:2 m above ground:",
                            #":PRES:surface::",
                            "TMP:surface:",
                            "DPT:2 m above ground:"]
    for tag in tags:
        tag_df=H.inventory(tag)
        tag_df = tag_df.reset_index(drop=True)
        print(len(tag_df))
        tag_match = [x for x in specific_tag_names if 
                     tag_df['variable'].iloc[0] in x][0]
        index_val = tag_df[tag_df['search_this'].str.contains(tag_match)].iloc[0].name
        ds = H.xarray(tag, remove_grib=True)
        predictions = list()
        for point in points:
            if "d2m" in ds:
                pred = ds['d2m'].sel(longitude=point[0],
                                     latitude=point[1],
                                     method='nearest').values.reshape(1)
            elif "tcc" in ds:
                pred = ds['tcc'].sel(longitude=point[0],
                                     latitude=point[1],
                                     method='nearest').values.reshape(1)
            elif "u" in ds:
                pred = ds['u'].sel(longitude=point[0],
                                   latitude=point[1],
                                   method='nearest').values.reshape(1)
            elif "v" in ds:
                pred = ds['v'].sel(longitude=point[0],
                                   latitude=point[1],
                                   method='nearest').values.reshape(1)
            elif "r2" in ds:
                pred = ds['r2'].sel(longitude=point[0],
                                    latitude=point[1],
                                    method='nearest').values.reshape(1)
            elif "sp" in ds:
                pred = ds['sp'].sel(longitude=point[0],
                                    latitude=point[1],
                                    method='nearest').values.reshape(1)
            elif "t" in ds:
                pred = ds['t'].sel(longitude=point[0],
                                   latitude=point[1],
                                   method='nearest').values.reshape(1)
            else:
                break
            predictions.append(pred)
        predictions = list(np.concatenate(predictions))
        pred_df = pd.DataFrame()
        pred_df['longitude'] = [point[0] for point in points]
        pred_df['latitude'] = [point[1] for point in points]
        pred_df['forecast_time'] = date
        pred_df['time_horizon_hrs'] = time_horizon
        pred_df['tag'] = tag
        pred_df['value'] = predictions
    return pred_df



forecast_dir = "C:/Users/kperry/data/"
if __name__ == "__main__":
    df = pd.read_csv("nodes.csv")
    df = df[df['geocode'].isin(['9q9wtp', '9q6tde', '9qcbq0', '9qcf5u', '9q97v8', '9q9hq4'])]
    points = [(y,x) for x,y in zip(df['Lat'], df['Long'])]
    names = list(df['county'])
    # Associated date range for the forecasts
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="6H")
    master_prediction_df = pd.DataFrame()
    # Do HRR up to 18 hours first (2 hour forecasts)
    for date in dates:
        for time_horizon in range(1, 19, 1):
            print(time_horizon)
            hrrr_pred_df = delayed(pull_herbie_hrr_data)(date, time_horizon).compute()
            hrrr_pred_df.to_csv(os.path.join("C:/Users/kperry/Documents/herbie_forecasts",
                                             date.strftime("%Y-%m-%d") + "_" + str(time_horizon) + "hr.csv"
                                             ), index=False)
            # Delete all of the accumulated grib2 files so we don't run out of storage
            #shutil.rmtree(forecast_dir)
        gefs_time_horizons = [*range(24,78, 6)]
        for time_horizon in gefs_time_horizons:
            gefs_pred_df = delayed(pull_herbie_gefs_data)(date, time_horizon).compute()
            gefs_pred_df.to_csv(os.path.join("C:/Users/kperry/Documents/herbie_forecasts",
                                             date.strftime("%Y-%m-%d") + "_" + str(time_horizon) + "hr.csv"
                                             ), index=False)
            #shutil.rmtree(forecast_dir)