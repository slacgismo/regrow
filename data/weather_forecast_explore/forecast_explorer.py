# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                                     main.py
#
# 
#              -------------------
# author 	         : Robert White
# date         	     : Oct 22, 2024
# copyright       : (C) 2024 by  NREL
# email              : robert.white@nrel.gov
# //////////////////////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////////////////////
# UPDATES:
# //////////////////////////////////////////////////////////////////////////////
                                                                                                          
# //////////////////////////////////////////////////////////////////////////////
# This file is part of the ReGROW project Scripts.
#
# The ReGROW scripts are free software: 
# you can redistribute it and/or modify it under the terms 
# of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# The ReGROW scripts are distributed in the 
# hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICU-
# LAR PURPOSE.  See the GNU General Public 
# License for more details.
#
# a copy of the GNU General Public License should a 
# company the ReGROW scripts. on release. 
# If not, see <http:#www.gnu.org/licenses/>.
#
# We also ask that you maintain the authorship block and 
# that any publications surrounding, attributed to, or linked 
# to this file or entire software system are also credited to 
# the authors and institution of this software
#
from datetime import datetime, date, timedelta
import argparse
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import boto3
import re
import glob
import seaborn as sns
import os 
from PIL import Image
from pytz import timezone
import support as sp

#-----------------------------------------------------------------------------------------------------
def convert_timestamp(ts):
    '''
    Split the string to isolate the date part
    
    Parameters
    --------------------------
    ts : string
         The original string of a timestamp containing data and time
     
    Returns
    -------------------
    date_part : string
         The single date element of a timestamp
    '''
    # 
    date_part = ts.split("T")[0] + " " + ts.split("T")[1][:8]
    return date_part


#-----------------------------------------------------------------------------------------------------
def create_composite_plots():
    '''
    format helping method to creates a single png by vertically stacking three other pngs.
    '''
    # Open each of your PNG images
    img1 = Image.open('file_1.png')
    img2 = Image.open('file_2.png')
    img3 = Image.open('file_3.png')
    
    # Make a list of these images
    images = [img1, img2, img3]
    
    # Assume all images have the same width
    # If they differ, you could choose the max width or handle resizing
    width = images[0].width
    
    # Calculate the total height needed
    total_height = sum(img.height for img in images)
    
    # Create a new blank image with the combined height
    combined_img = Image.new('RGB', (width, total_height))
    
    # Paste each image into the new image, one below the other
    y_offset = 0
    for img in images:
        combined_img.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save the final combined image
    combined_img.save("path/file_to_write.png")
    return

#-----------------------------------------------------------------------------------------------------
def animate_plots():
    '''
    Generates an animated gif from a series of filenames
    '''
    # List of image filenames in the order you want them animated
    image_files = [
        "plot_file_1.png",
        "plot_file_2.png",
        "plot_file_3.png",
        "plot_file_4.png",
        ]
      
    # Open each file and append to frames list
    frames = []
    for file in image_files:
        new_frame = Image.open('plots/' + file)
        frames.append(new_frame)
    
    # Save the first frame and append the rest
    # duration: time in ms for each frame
    # loop=0: infinite loop
    frames[0].save(
        "path/file_to_write.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1000,  # 1000 ms = 1 second per frame
        loop=10
    )
    return





#-----------------------------------------------------------------------------------------------------
def plot_scatters(df, target_dir):
    '''
    
    '''
    # Scatter plot of distance vs. temperature difference
    # Filter the DataFrame to remove outliers (diff > +20 or diff < -20)
    filtered_df = df[(df['diff_temp_max'] > -20) & (df['diff_temp_max'] < 20)]
    filtered_df = filtered_df[(filtered_df['diff_temp_min'] > -20) & (filtered_df['diff_temp_min'] < 20)]
    filtered_df = filtered_df[(filtered_df['diff_temp_mean'] > -20) & (filtered_df['diff_temp_mean'] < 20)]

    #Weather Station diff
    #plt.figure(figsize=(8, 6))
    ##sns.scatterplot(x=filtered_df['distance_ws_to_node_km'], y=filtered_df['diff_temp_max'], alpha=0.7)
    #sns.lmplot(data=filtered_df, x='distance_ws_to_node_km', y='diff_temp_max', aspect=2, scatter_kws={'alpha':0.5})
    #plt.title('Weater Station Distance vs. Temperature Max Difference')
    #plt.xlabel('Distance (km)')
    #plt.ylabel('Temperature Difference (°C)')
##    plt.show()
    #plt.tight_layout()
    #plt.savefig(target_dir + '/ow_master_max_temperature_dist_trend.png', bbox_inches='tight', dpi=300)  

    #plt.figure(figsize=(8, 6))
    ##sns.scatterplot(x=filtered_df['distance_ws_to_node_km'], y=filtered_df['diff_temp_max'], alpha=0.7)
    #sns.lmplot(data=filtered_df, x='distance_ws_to_node_km', y='diff_temp_min', aspect=2, scatter_kws={'alpha':0.5})
    #plt.title('Weater Station Distance vs. Temperature Min Difference')
    #plt.xlabel('Distance (km)')
    #plt.ylabel('Temperature Difference (°C)')
##    plt.show()
    #plt.tight_layout()
    #plt.savefig(target_dir + '/ow_master_min_temperature_dist_trend.png', bbox_inches='tight', dpi=300)  

    #plt.figure(figsize=(8, 6))
    ##sns.scatterplot(x=filtered_df['distance_ws_to_node_km'], y=filtered_df['diff_temp_max'], alpha=0.7)
    #sns.lmplot(data=filtered_df, x='distance_ws_to_node_km', y='diff_temp_mean', aspect=2, scatter_kws={'alpha':0.5})
    #plt.title('Weater Station Distance vs. Temperature Mean Difference')
    #plt.xlabel('Distance (km)')
    #plt.ylabel('Temperature Difference (°C)')
    ##plt.show()
    #plt.tight_layout()
    #plt.savefig(target_dir + '/ow_master_mean_temperature_dist_trend.png', bbox_inches='tight', dpi=300)  

    
    std_devs_max = filtered_df.groupby('day_diff')['diff_temp_max'].std()
    std_devs_min = filtered_df.groupby('day_diff')['diff_temp_min'].std()
    std_devs_mean = filtered_df.groupby('day_diff')['diff_temp_mean'].std()
    # Calculate mean and standard deviation for each group
    #stats_max = df.groupby('day_diff')['diff_temp_max'].agg(['mean', 'std', 'max'])
 
    #day Diff
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=filtered_df, x='day_diff', y='diff_temp_max',  hue='day_diff', palette='Reds', legend=False)#palette='coolwarm', showfliers=True)
    ymax = ax.get_ylim()[1]  
    # Add standard deviation annotations
    for i, std in enumerate(std_devs_max):
        plt.text(x=i+0.25, 
                 y=ymax - 1.0,  # Position above the plot
                 s=f"σ={std:.2f}", 
                 horizontalalignment='center', 
                 fontsize=10, 
                 color='black')
            
    #for i, (mean, std) in enumerate(zip(stats_max['mean'], stats_max['std'])):
        #plt.errorbar(x=i, y=mean, yerr=std, fmt='o', color='black', capsize=5)    
        
    # Add titles and labels
    plt.title('Max Temperature Difference  Between Forecast and Actual by Forecast Deviation', fontsize=16)
    plt.xlabel('Forecast Days (delta day)', fontsize=14)
    plt.ylabel('Temperature Difference (C)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)  
    #plt.show()
    plt.tight_layout()
    plt.savefig(target_dir + '/ow_master_max_temperature_dev_box.png', bbox_inches='tight', dpi=300)  

    #minimum
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=filtered_df, x='day_diff', y='diff_temp_min', hue='day_diff', palette='Blues', legend=False)
    ymax = ax.get_ylim()[1]  
    # Add standard deviation annotations
    for i, std in enumerate(std_devs_min):
        plt.text(x=i+0.25, 
                 y=ymax - 1.0,  # Position above the plot
                 s=f"σ={std:.2f}", 
                 horizontalalignment='center', 
                 fontsize=10, 
                 color='black')

    # Add titles and labels
    plt.title(' Min Temperature Difference Between Forecast and Actual by Forecast Deviation', fontsize=16)
    plt.xlabel('Forecast Days (delta day)', fontsize=14)
    plt.ylabel('Temperature Difference (C)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)  
    #plt.show()
    plt.tight_layout()
    plt.savefig(target_dir + '/ow_master_min_temperature_dev_box.png', bbox_inches='tight', dpi=300)  

    #Mean
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=filtered_df, x='day_diff', y='diff_temp_mean', hue='day_diff', palette='Purples', legend=False)
    ymax = ax.get_ylim()[1]  
    # Add standard deviation annotations
    for i, std in enumerate(std_devs_mean):
        plt.text(x=i+0.25, 
                 y=ymax - 1.0,  # Position above the plot
                 s=f"σ={std:.2f}", 
                 horizontalalignment='center', 
                 fontsize=10, 
                 color='black')

    # Add titles and labels
    plt.title('Mean Temperature Difference Between Forecast and Actual by Forecast Deviation' , fontsize=16)
    plt.xlabel('Forecast Days (delta day)', fontsize=14)
    plt.ylabel('Temperature Difference (C)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)  
    #plt.show()
    plt.tight_layout()
    plt.savefig(target_dir + '/ow_master_mean_temperature_dev_box.png', bbox_inches='tight', dpi=300)  
    
    return

#-----------------------------------------------------------------------------------------------------
def plot_dist_forecast_to_actual_diffs(target_dir, node_id, df):
    '''
    
    '''
    # Filter the DataFrame to remove outliers (diff > +20 or diff < -20)
    filtered_df = df[(df['diff_temp_max'] > -20) & (df['diff_temp_max'] < 20)]
    filtered_df = filtered_df[(filtered_df['diff_temp_min'] > -20) & (filtered_df['diff_temp_min'] < 20)]
    filtered_df = filtered_df[(filtered_df['diff_temp_mean'] > -20) & (filtered_df['diff_temp_mean'] < 20)]
       
    plt.figure(figsize=(14, 6))
    sns.histplot(filtered_df, x='diff_temp_max', hue='day_diff', palette='Reds', kde=True, bins=30)
    plt.title('Variation in Max Temperature Differences Across Forecast and Actual Date Deviations ')
    plt.xlabel('Temperature Difference (°C)')
    plt.ylabel('Frequency')
    #plt.show()
    # Save the plot as  file 
    plt.tight_layout()
    if node_id:
        plt.savefig(target_dir + '/' + node_id + '_max_temperature_difference_distribution.png', bbox_inches='tight', dpi=300)  
    else:
        plt.savefig(target_dir + '/ow_master_max_temperature_difference_distribution.png', bbox_inches='tight', dpi=300)  
        
    
    plt.figure(figsize=(14, 6))
    sns.histplot(filtered_df, x='diff_temp_min', hue='day_diff', palette='Blues', kde=True, bins=30)
    plt.title('Variation in Min Temperature Differences Across Forecast and Actual Date Deviations ')
    plt.xlabel('Temperature Difference (°C)')
    plt.ylabel('Frequency')
    #plt.show()
    # Save the plot as  file 
    plt.tight_layout()
    if node_id:
        plt.savefig(target_dir + '/' + node_id + '_min_temperature_difference_distribution.png', bbox_inches='tight', dpi=300)  
    else:
        plt.savefig(target_dir + '/ow_master_min_temperature_difference_distribution.png', bbox_inches='tight', dpi=300)  

    
    plt.figure(figsize=(14, 6))
    sns.histplot(filtered_df, x='diff_temp_mean', hue='day_diff', palette='Purples', kde=True, bins=30)
    plt.title('Variation in Mean Temperature Differences Across Forecast and Actual Date Deviations ')
    plt.xlabel('Temperature Difference (°C)')
    plt.ylabel('Frequency')
    #plt.show()    
    # Save the plot as  file 
    plt.tight_layout()
    if node_id:
        plt.savefig(target_dir + '/' + node_id + '_mean_temperature_difference_distribution.png', bbox_inches='tight', dpi=300)  
    else:
        plt.savefig(target_dir + '/ow_master_mean_temperature_difference_distribution.png', bbox_inches='tight', dpi=300)  

  
    return


#------------------------------------------------------------------------------------------------------
def merge_nsrdb(node_id, df, df_nsrdb, conus=False):
    '''
    Merges the exisitng forecast and actual file with nsrdb file for node.
    Calculates the cloud coverage for the node using an approximation
    based on (dhi/(dhi+dni))*100, to give a percentage cloud cover.
    Parameters
    --------------------------
    node_id : str
         WEC node identifier
    df : dataframe object
         previously merged forecast (openWeather) and actual weather
         (NOAA) csv file as a pandas dataframe
    df_nsrdb : dataframe object
         WECC node level 
     
    Returns
    -------------------
    date_part : string
         The single date element of a timestamp
    '''
    if conus:
        df_nsrdb.rename(columns={
            df_nsrdb.columns[0]: "timestamp",  
            'Temperature' : 'temp_air',
            'DHI' : 'dhi',
            'DNI' : 'dni',
            'GHI' : 'ghi',
            'Wind Speed' : 'wind_speed',
            'Air Temperature at hub height (°C)' : 'hub_air_temp',
            'Surface Air Pressure (Pa)' : 'air_pressure',
            'Wind Speed at hub height (m/s)' : 'hub_wind_speed',
            'Wind Direction at hub height(°)' : 'hub_wind_dir'        
            }, inplace=True)
    else:
        df_nsrdb.rename(columns={
            df_nsrdb.columns[0]: "timestamp",  
            'Temperature' : 'temp_air',
            'DHI' : 'dhi',
            'DNI' : 'dni',
            'GHI' : 'ghi',
            'Wind Speed' : 'wind_speed',
            }, inplace=True)

    # Ensure timestamp column is datetime and timezone-aware
    df_nsrdb["timestamp"] = pd.to_datetime(df_nsrdb["timestamp"])
    
    # Filter to desired range
    start = pd.Timestamp("2018-01-01 00:00:00", tz="US/Pacific")
    end   = pd.Timestamp("2022-12-31 23:59:59", tz="US/Pacific")
    df_nsrdb_filtered = df_nsrdb[(df_nsrdb["timestamp"] >= start) & (df_nsrdb["timestamp"] <= end)]
    
    # Ensure df['forecast_time'] is also datetime and timezone-aware
    #df["forecast_time"] = pd.to_datetime(df["forecast_time"], errors="coerce")
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)   
    
    # Now check if it's tz-aware and convert/localize as needed
    if df["forecast_time"].dt.tz is not None:
        df["forecast_time"] = df["forecast_time"].dt.tz_convert("US/Pacific")
    else:
        df["forecast_time"] = df["forecast_time"].dt.tz_localize("US/Pacific", ambiguous='NaT', nonexistent='shift_forward')
        
    # Prefix all columns in df_nsrdb except timestamp
    df_nsrdb_filtered = df_nsrdb_filtered.rename(columns={col: f"nsrdb_{col}" for col in df_nsrdb_filtered.columns if col != "timestamp"})
    if conus:
        df_nsrdb_filtered.rename(columns={ 
            "nsrdb_hub_air_temp" : "hub_air_temp", 
            "nsrdb_hub_air_pressure" : "hub_air_pressure", 
            "nsrdb_hub_wind_speed" : "hub_wind_speed" , 
            "nsrdb_hub_wind_dir" : "hub_wind_dir" 
        }, inplace=True)
            
    # Merge on matching timestamps
    merged_df = pd.merge(df, df_nsrdb_filtered, how="left",
                         left_on="forecast_time", right_on="timestamp")
    
    # Drop the now-duplicate timestamp column if desired
    merged_df.drop(columns=["timestamp"], inplace=True)
    
    #Calculate out the nsrdb_cloud_coverage using dhi to dni ratios
    merged_df["nsrdb_calc_cloud_coverage"] = merged_df.apply(
        lambda row: round(((row["nsrdb_dhi"] / (row["nsrdb_dhi"] + row["nsrdb_dni"])) * 100), 4)
        if pd.notnull(row["nsrdb_dhi"]) and pd.notnull(row["nsrdb_dni"]) and (row["nsrdb_dhi"] + row["nsrdb_dni"]) > 0
        else None,
        axis=1
    )
    #Write file out to temp location
    #merged_df.to_csv('C:/Regrow/Historic_weather_data/NSRDB_Target_nodes/merged_' + node_id + '_forecast_actual_nsrdb.csv')
    return merged_df


#------------------------------------------------------------------------------------------------------
def cloud_coverage_comparison(df):
    '''
    Plots the comparison of NOAA vs NSRDB
    Parameters
    --------------------------
    df : dataframe object
         previously merged forecast (openWeather) and actual weather
         (NOAA) and NSRDB values
     
    Results
    -------------------
    Plot comparison of NSRDB and NOAA
    '''
    
    # Filter rows with valid nsrdb coverage
    node_id =df["node_id"].iloc[0]
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
    df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
    start_date = pd.Timestamp("2020-07-25 00:00:00").tz_localize('UTC') 
    end_date = pd.Timestamp("2020-08-15 23:59:59").tz_localize('UTC') 
    df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]

    #filtered_df = df[df["nsrdb_calc_cloud_coverage"].notnull()]
    filtered_df = df[df["nsrdb_wind_speed"].notnull()]
    #filtered_df = df[df["nsrdb_temp_air"].notnull()]
    #Calc Rsquare
    #r2 = r2_score( filtered_df["nsrdb_calc_cloud_coverage"], filtered_df["clouds"])
    r2 = r2_score( filtered_df["nsrdb_wind_speed"], filtered_df["wind_speed"])
    #r2 = r2_score( filtered_df["nsrdb_temp_air"], filtered_df["temperature"])
    
    # Fit trend line: forecast = slope * actual + intercept
    slope, intercept, r_value, p_value, std_err = linregress(filtered_df["nsrdb_wind_speed"],  filtered_df["wind_speed"])
    trend_line = slope * filtered_df["nsrdb_wind_speed"] + intercept   
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    #plt.scatter(
        #filtered_df["clouds"],
        #filtered_df["a_cloud_cover"],
        #alpha=0.5,
        #color="steelblue"
    #)
    sc = plt.scatter(
        filtered_df["nsrdb_wind_speed"],
        filtered_df["wind_speed"],
        alpha=0.5,
        c=filtered_df['hour_diff'],
        cmap='cividis',
        edgecolor='none',
        s=30
    )
    
    # Add colorbar (gradient legend)
    cbar = plt.colorbar(sc)
    cbar.set_label('Forecast Hour Delta')

    # Plot 1:1 line
    min_val = min(filtered_df["nsrdb_wind_speed"].min(), filtered_df["wind_speed"].min())
    max_val = max(filtered_df["nsrdb_wind_speed"].max(), filtered_df["wind_speed"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5)
    # Plot trend line
    plt.plot(filtered_df["nsrdb_wind_speed"], trend_line, 'g-', color='darkorange', label=f'Trend Line: y={slope:.2f}x+{intercept:.2f}')    
    
    # Labels and formatting
    #plt.xlabel("Forecast Cloud Coverage (%)")
    #plt.ylabel("NOAA Actual Cloud Coverage (%)")
    #plt.title("OW Forecast vs NOAA Cloud Coverage")
    plt.xlabel("NSRDB Wind Speed (m/s)")
    plt.ylabel("Forecast Wind Speed (m/s)")
    plt.title(node_id + " - OpenWeather Forecast vs NSRDB  Wind Speed (1 hour intervals)")
    # Add R-squared text to plot
    plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))    
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
    return

#------------------------------------------------------------------------------------------------------
def cloud_coverage_comparison_other(df_merged):
    '''
    Merges the exisitng forecast and actual file with nsrdb file for node.
    Calculates the cloud coverage for the node using an approximation
    based on (dhi/(dhi+dni))*100, to give a percentage cloud cover.
    Parameters
    --------------------------
    df_merged : dataframe object
         previously merged forecast (openWeather) and actual weather
         (NOAA) and NSRDB values
     
    Results
    -------------------
    Plot comparison of NSRDB and NOAA
    '''
    # Filter rows with valid nsrdb coverage
    df = df_merged[df_merged["nsrdb_calc_cloud_coverage"].notnull()]

    df["residual"] = df["a_cloud_cover"] - df["nsrdb_calc_cloud_coverage"]
    df["forecast_hour"] = df["forecast_time"].dt.hour + df["forecast_time"].dt.minute / 60.0
    
    #Scatter  plot
    #plt.figure(figsize=(12, 6))
    #plt.scatter(df["forecast_hour"], df["residual"], alpha=0.4, color="darkorange")
    #plt.axhline(0, color="gray", linestyle="--")
    #plt.xlabel("Time of Day (Hour)")
    #plt.ylabel("Residual (NOAA - NSRDB)")
    #plt.title("Residuals by Forecast Time of Day")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
        
    #Histogram
    #plt.figure(figsize=(8, 5))
    #plt.hist(df["residual"].dropna(), bins=50, color="slateblue", edgecolor="black", alpha=0.7)
    #plt.axvline(0, color="gray", linestyle="--")
    #plt.xlabel("Residual (Actual - NSRDB)")
    #plt.ylabel("Frequency")
    #plt.title("Distribution of Cloud Coverage Residuals")
    #plt.tight_layout()
    #plt.show()
    
    #Box plot
    #df["forecast_hour"] = df["forecast_time"].dt.hour
    
    #plt.figure(figsize=(12, 6))
    #df.boxplot(column="residual", by="forecast_hour", grid=False)
    #plt.axhline(0, color="gray", linestyle="--")
    #plt.title("Residuals by Hour of Forecast Time")
    #plt.suptitle("")  # Removes default "Boxplot grouped by ..."
    #plt.xlabel("Hour of Day")
    #plt.ylabel("Residual (Actual - NSRDB)")
    #plt.tight_layout()
    #plt.show()
    
    
    #Join Density Plot
    # Optional: filter a time range (e.g., 6 AM to 6 PM)
    #df_filtered = df[(df["forecast_hour"] >= 6) & (df["forecast_hour"] <= 18)]
    # Or just use all:
    df_filtered = df.copy()
    
    sns.jointplot(
        data=df_filtered,
        x="nsrdb_calc_cloud_coverage",
        y="a_cloud_cover",
        kind="kde",  # use 'scatter' if you prefer
        fill=True,
        color="teal"
    )
    
    
    #Examine direct comparison (NOTE: results not conclusive)
    # Create scatter plot
    #plt.figure(figsize=(10, 6))
    #plt.scatter(
        #filtered_df["nsrdb_calc_cloud_coverage"],
        #filtered_df["a_cloud_cover"],
        #alpha=0.5,
        #color="steelblue"
    #)
    
    ## Plot 1:1 line
    #min_val = min(filtered_df["nsrdb_calc_cloud_coverage"].min(), filtered_df["a_cloud_cover"].min())
    #max_val = max(filtered_df["nsrdb_calc_cloud_coverage"].max(), filtered_df["a_cloud_cover"].max())
    #plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5)
    
    ## Labels and formatting
    #plt.xlabel("NSRDB Calculated Cloud Coverage (%)")
    #plt.ylabel("NOAA Cloud Coverage (%)")
    #plt.title("NSRDB vs NOAA Cloud Coverage with 1:1 Line")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
    
    
    
    
    return



#-----------------------------------------------------------------------------------------------------
def compare_linear_forecast_model_actual(node_id,
                                         df, 
                                         period,
                                         start_timestamp = '2020-08-01 00:00:00',
                                         end_timestamp= '2020-08-23 23:59:59', 
                                         norm_frame='',
                                         data_type='temperature',
                                         box=False,
                                         target_dir = '',
                                         show_forecast_scatter=False, 
                                         show_forecast_median=False, 
                                         show_noaa=False,
                                         show_nsrdb=False,
                                         show_conus=False, 
                                         ):
    '''
    Create a complex line and scatter plot of the forecast, actual and model datasets as a comparison
    to their behaviors for a type of forecast (temp, wind, etc.)
    
    Parameters
    -------------------------------------------
    node_id : str
         WECC Node id defined as partof REGROW Project.
    df : dataframe object
         Composite dataframe generated from forecast, actual and model data
    period : int
          Forecast period. Used to provide full resolution or daily desolutions of the
          data. Calls function to do averaged roll ups for the period. Period of 1 is one
          hour and is native resolution of the data frame.
    start_timestamp : str
          The beginning timestamp of data to bracket the analysis. ISO 8601-format,
          with space seperator. Default  2020-08-01 00:00:00
    end_timestamp : str
          The beginning timestamp of data to bracket the analysis. ISO 8601-format,
          with space seperator. Default  2020-08-23 23:59:59
    data_type : str
          Which of the data types is the plotting to be done for. Possible values of
          temperature, wind_speed, and clouds
    box : bool
         Create box and whisker plot or not.
    target_dir : str
         Path to store a copy of the results to. Filename is predefined. Default is empty
         and if empty, no file is written.
    show_forecast_scatter : bool
         Show the forecast values as a scatter plot, with hue of markers colored by
         distance of forecast from prediction.
         
    Results
    -------------------------------------------
    Plot of values from matplotlib package.
        
    '''
    if period:
        forecast_period = period
        #Begin rollingup forecast data by blocks basedon forecast_period
        if forecast_period > 24 or forecast_period < 1:
            raise ValueError("forecast_period must be an integer between 1 and 24 hours.")
    else:
        forecast_period = 1

    #select target fields:
    if data_type == 'temperature':
        forecast = 'temperature'
        actual = 'a_temperature'
        model = 'nsrdb_temp_air'
        hub = 'hub_air_temp'
        y_label = "Temperature (C)"
        title = node_id + " Forecasted vs Observed and Modeled Temperature\nColored by Forecast Lead Time (" + str(forecast_period) + "H resampling)"
        filename = node_id + "_forecast_actual_modeled_temp_" + str(forecast_period) + "h.png"
    elif  data_type == 'wind_speed':
        forecast = 'wind_speed' 
        actual = 'a_wind_speed'
        model = 'nsrdb_wind_speed'
        hub = 'hub_wind_speed'
        y_label = "Wind Speed (m/s)"
        title = node_id + " Forecasted vs Observed and Modeled Wind Speed\nColored by Forecast Lead Time (" + str(forecast_period) + "H resampling)"
        filename = node_id + "_forecast_actual_modeled_ws_" + str(forecast_period) + "h.png"
    elif  data_type == 'clouds':
        actual = 'a_cloud_cover'
        model = 'nsrdb_calc_cloud_coverage'
        y_label = "Cloud Cover (%)"
        title = " Forecasted vs Observed and Modeled Cloud Cover\nColored by Forecast Lead Time (" + str(forecast_period) + "H resampling)"
        filename = node_id + "_forecast_actual_modeled_clouds_" + str(forecast_period) + "h.png"
    else:
        print ('Unknown data type for comparison. Exiting')
        return

    #Forecast vs actual vas model
    #Prep and filter
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
    df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
    
    #Define date range of data to filter to
    start_date = pd.Timestamp(start_timestamp).tz_localize('UTC') 
    end_date = pd.Timestamp(end_timestamp).tz_localize('UTC') 
    df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]
    
    #For forecasts
    df_multi = df.set_index(['predict_time', 'forecast_time'])

    # Sort by forecast_time and predict_time so earliest predictions come first
    df = df.sort_values(by=['forecast_time', 'predict_time'])
    # Sort and prepare main plot data (first forecasts only)
    first_forecast_df = df.sort_values(by=['forecast_time', 'predict_time']) \
                          .drop_duplicates(subset='forecast_time', keep='first') \
                          .set_index('forecast_time') \
                          .sort_index()
    
    # Prepare scatter data
    scatter_df = df[['forecast_time', data_type, 'hour_diff']].dropna()
    scatter_df['forecast_time'] = pd.to_datetime(scatter_df['forecast_time'])
    
    #print('df size ' + str(df.shape))
    # Resample within each predict_time group
    df_resampled = (
        df_multi
        .groupby(level='predict_time')
        .resample(str(forecast_period) + 'h', level='forecast_time')  # resample forecast_time within each predict_time
        .agg({
            data_type : 'mean',       # or 'first', 'median', etc.
            'hour_diff': 'first',       # preserve the correct hour_diff for scatter color
        })
        .dropna()
        .reset_index()
    )
    # Calculate median forecast trend from resampled scatter data
    median_forecast = (
        df_resampled
        .groupby('forecast_time')[data_type]
        .median()
        .reset_index()
    )    
    #print('resampled size ' + str(df_resampled.shape))
    
    #Cretating boxplot dataset
    # Group wind_actual_resampled by forecast_time
    boxplot_data = (
        df_resampled
        .groupby('forecast_time')[data_type]
        .apply(list)  # create list of forecast values per time
    )    
        
    if forecast_period > 1:
        # Resample actual (NOAA) and modeled (NSRDB) using same forecast_period
        first_forecast_resampled = (
            first_forecast_df
            .resample(str(forecast_period) + 'h')  # resampling by forecast_period hours
            .agg({
                'predict_time': 'first',
                forecast : 'mean',
                actual : 'mean',    # or 'first' depending on what you want
                model : 'mean',
                hub: 'mean'
            })
            .dropna()
            .reset_index()
        )    
    
    # Plot
    plt.figure(figsize=(16, 9), dpi=60)  
    
    #if wanting to do box and whiskers
    if box:
        # Create x positions for boxplots
        positions = mdates.date2num(boxplot_data.index)
        
        box_width_days = forecast_period / 24.0  # since 1 day = 24 hours
        
        # Slightly smaller to avoid overlap, e.g. 80% of width
        box_width_days *= 0.8    
    
        # Draw boxplots
        plt.boxplot(boxplot_data.values, positions=positions, widths=box_width_days, patch_artist=True,
                    boxprops=dict(facecolor='none', color='black', linewidth = 1.0),
                    medianprops=dict(color='black'))
    #End box plotting
    
    #Check for perodicity. If greater than 1, resample.
    if forecast_period > 1:
        if show_forecast_scatter:
            # Scatter: forecast colored by hour_diff
            sc = plt.scatter(
                df_resampled['forecast_time'],
                df_resampled[data_type],
                c=df_resampled['hour_diff'],
                cmap='cividis',
                alpha=0.7,
                label='Forecast'
            )
        # Line plot actual NOAA (resampled)
        if show_noaa:
            plt.plot(first_forecast_resampled['forecast_time'], first_forecast_resampled[actual],
                     color='#00BCD4', label='NOAA (actual)', linewidth=2)
        
        if show_nsrdb:
            # Line plot: NSRDB Modeled (resampled)
            plt.plot(first_forecast_resampled['forecast_time'], first_forecast_resampled[model],
                     color='#D55E00', label='NSRDB (modeled)', linewidth=2)        
    
        # Line plot: Conus Modeled (resampled)
        if show_conus and data_type != 'clouds':
            plt.plot(first_forecast_resampled['forecast_time'], first_forecast_resampled[hub],
                     color='#800000', label='Conus (modeled, hub height)', linewidth=2)
        
        if show_forecast_median:
            # Median trend line for forecast values
            plt.plot(median_forecast['forecast_time'], first_forecast_resampled[forecast],
                     color='#CC79A7', linestyle='--', linewidth=2, label='Forecast Median')   

    else: # For standard file rate of hourly
        if show_forecast_scatter:
            # Scatter: forecast colored by hour_diff
            sc = plt.scatter(
                scatter_df['forecast_time'],
                scatter_df[data_type],
                c=scatter_df['hour_diff'],
                cmap='cividis',
                alpha=0.7,
                label='Forecast '
            )
        # Line plot actual NOAA (first forecast)
        if show_noaa:
            plt.plot(first_forecast_df.index, first_forecast_df[actual],
                 color='#00BCD4', label='NOAA (actual)', linewidth=2)
        
        # Line plot:  NSRDB Modeled nsrdb_wind_speed
        if show_nsrdb:
            plt.plot(first_forecast_df.index, first_forecast_df[model],
                 color='#D55E00', label='NSRDB (modeled)', linewidth=2)
    
        # Line plot: Conus Modeled (resampled)
        if show_conus and data_type != 'clouds':
            plt.plot(first_forecast_df.index, first_forecast_df[hub],
                     color='#800000', label='Conus (modeled, hub height)', linewidth=2)
            
        # Median trend line for forecast values
        if show_forecast_median:
            plt.plot(first_forecast_df.index, median_forecast[data_type],
                 color='#CC79A7', linestyle='--', linewidth=2, label='Forecast Median')   
           
    if show_forecast_scatter:
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Forecast Hour Difference')
    
    # Labels and legend
    plt.xlabel("Forecast Time")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    #Adjust x axis lables if boxing.
    if box:
        # Format x-axis as datetime
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.xticks(rotation=45)
    
    if target_dir:
        plt.savefig(target_dir + filename)
    plt.show()
    
    if forecast_period > 1:
        return first_forecast_resampled
    else:
        return first_forecast_df

#-----------------------------------------------------------------------------------------------------
def normalize_linear_ground_vs_hub(node_id,
                                   df,
                                   norm_df, 
                                   start_timestamp = '2020-08-01 00:00:00',
                                   end_timestamp= '2020-08-23 23:59:59', 
                                   data_type='wind_speed',
                                   target_dir = '', 
                                   line_plot = False, 
                                   regress_plot = False):
    '''
    normalizes the hub height and actual (ground) environmental signals. Then will
    plot the linear comparion and/or a linear regression of the values.
    
    Parameters
    -------------------------------------------
    node_id : str
         WECC Node id defined as partof REGROW Project.
    df : dataframe object
         Composite dataframe generated from forecast, actual and model data
    normalizes : dataframe object
          If passed in and not default empty string, then it is used instead of computing a new frame.
    start_timestamp : str
          The beginning timestamp of data to bracket the analysis. ISO 8601-format,
          with space seperator. Default  2020-08-01 00:00:00
    end_timestamp : str
          The beginning timestamp of data to bracket the analysis. ISO 8601-format,
          with space seperator. Default  2020-08-23 23:59:59
    data_type : str
          Which of the data types is the plotting to be done for. Possible values of
          temperature, wind_speed, and clouds
     target_dir : str
         Path to store a copy of the results to. Filename is predefined. Default is empty
         and if empty, no file is written.
    line_plot : bool
         Show a plot of the new normalized data. Default False
    regress_plot : bool
          Show a scatter plot and regression measurments. Default False.
   
    Results
    -------------------------------------------
    normalized_df : returns the normalized data frame.
        
    '''
    #select target fields:
    if data_type == 'temperature':
        actual = 'a_temperature'
        hub = 'hub_air_temp'
        norm_actual =  'a_temperature_scaled'
        norm_hub = 'hub_air_temp_scaled'
        y_label = "Normalized Temperature"
        title = node_id + " Normalized Observed and Modeled Temperature"
        filename = node_id + "_normalized_actual_modeled_temp.png"
    elif  data_type == 'wind_speed':
        actual = 'a_wind_speed'
        hub = 'hub_wind_speed'
        norm_actual =  'a_wind_speed_scaled'
        norm_hub = 'hub_wind_speed_scaled'
        y_label = "Normalized Wind Speed"
        title = node_id + " Normalized Observed and Modeled Wind Speed"            
        filename = node_id + "_normalized_actual_modeled_ws.png"
    elif  data_type == 'clouds':
        print ('Hub has no different cloud comparison. Exiting')
        return
    else:
        print ('Unknown data type for comparison. Exiting')
        return
    
    #reset index
    df = df.reset_index()
    if not norm_df.empty:
        df = norm_df
        
    else:  #Create new normalized data frame
        scaler = MinMaxScaler()
        df[[norm_actual, norm_hub]] = scaler.fit_transform(df[[actual, hub]])

       #Prep and filter
        df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
        df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
        
        #Define date range of data to filter to
        start_date = pd.Timestamp(start_timestamp).tz_localize('UTC') 
        end_date = pd.Timestamp(end_timestamp).tz_localize('UTC') 
        df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]
    
        # Sort by forecast_time and predict_time so earliest predictions come first
        df = df.sort_values(by=['forecast_time', 'predict_time'])
        # Sort and prepare main plot data (first forecasts only)
        first_forecast_df = df.sort_values(by=['forecast_time', 'predict_time']) \
                              .drop_duplicates(subset='forecast_time', keep='first') \
                              .set_index('forecast_time') \
                              .sort_index()

    #Calculate figures of merit for linear values of temeperature at low altitudes
    if data_type == 'temperature':
        # Compute Pearson correlation and R²
        pearson_corr, _ = pearsonr(df[norm_actual], df[norm_hub])
        r2 = r2_score(df[norm_actual], df[norm_hub])

    #Different merit scores for a wind which is power law
    elif data_type == 'wind_speed':
        # Define power-law model: v_80 = a * v_ground^b
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        # Fit the model
        x = df[actual].values
        y = df[hub].values
        params, _ = curve_fit(power_law, x, y)
        a, b = params
        
        # Predict using fitted model
        y_pred = power_law(x, a, b)
        
        # Compute Pearson correlation and R²
        pearson_corr, _ = pearsonr(y, y_pred)
        r2 = r2_score(y, y_pred)
        
    if line_plot:
        plt.figure(figsize=(16, 9), dpi=60)  
        
        plt.plot(first_forecast_df.index, first_forecast_df[norm_actual],
             color='#00BCD4', label='NOAA (actual, ground)', linewidth=2, )
        
        # Line plot: Conus hub height wind_speed
        plt.plot(first_forecast_df.index, first_forecast_df[norm_hub],
                 color='#800000', label='Conus(modeled, hub height)', linewidth=2)
    
        #Labels and legend
        plt.xlabel("Forecast Time")
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if target_dir:
            test = True
            #plt.savefig(target_dir + node_id + "normalized_" + data_type + "_conus.csv")
        plt.show()
    
    if regress_plot:                   
        plt.figure(figsize=(16, 9), dpi=60)  
        #build seaborn plot
        sns.scatterplot(
            x=df[norm_actual],
            y=df[norm_hub],
            data=df,
            #palette="Blues",
            s=30,
        )
        
        sns.regplot(
            x=df[norm_actual],
            y=df[norm_hub],
            data=df,
            scatter=False,  # Don't replot scatter points
            color="grey",
            #ci=None,  # Remove confidence intervals
        )
    
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
        
        if data_type == 'wind_speed':
            plt.text(0.05, 0.7,
                 f"Fitted power-law: v_80 = {a:.3f} * v_ground^{b:.3f}\nPearson r = {pearson_corr:.2f}\nR² = {r2:.2f}",
                 fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        elif data_type == 'temperature':
            plt.text(0.05, 0.7,
                 f"Pearson r = {pearson_corr:.2f}\nR² = {r2:.2f}",
                 fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))           
        
        plt.title(node_id + ' - Correlation of '+ y_label + ' for Modeled Hub Height vs. Measured Ground',  fontsize=16)
        plt.xlabel('Measured '+ y_label,  fontsize=14)
        plt.ylabel('Hub Height '+ y_label, fontsize=14)
        plt.tight_layout()
        #save
        if target_dir:
            test = True
            #plt.savefig( target_dir + node_id + "_" + data_type + "_regress_actual_vs_hub.png", dpi=200)
        plt.show()
    return df


#-----------------------------------------------------------------------------------------------------
def compare_linear_ground_vs_hub(node_id,
                                 df,
                                 period,
                                 start_timestamp = '2020-08-01 00:00:00',
                                 end_timestamp= '2020-08-23 23:59:59', 
                                 data_type='wind_speed',
                                 box=False,
                                 target_dir = '',
                                 show_scatter=False,
                                 show_nsrdb=False,
                                 scatter_measured=False):
    '''
    Create a complex line and scatter plot of the forecast, actual and model datasets as a comparison
    to their behaviors for a type of forecast (temp, wind, etc.)
    
    Parameters
    -------------------------------------------
    node_id : str
         WECC Node id defined as partof REGROW Project.
    df : dataframe object
         Composite dataframe generated from forecast, actual and model data
    period : int
          Forecast period. Used to provide full resolution or daily desolutions of the
          data. Calls function to do averaged roll ups for the period. Period of 1 is one
          hour and is native resolution of the data frame.
    start_timestamp : str
          The beginning timestamp of data to bracket the analysis. ISO 8601-format,
          with space seperator. Default  2020-08-01 00:00:00
    end_timestamp : str
          The beginning timestamp of data to bracket the analysis. ISO 8601-format,
          with space seperator. Default  2020-08-23 23:59:59
    data_type : str
          Which of the data types is the plotting to be done for. Possible values of
          temperature, wind_speed, and clouds
    box : bool
         Create box and whisker plot or not.
    show_scatter : bool
         Show the forecast values as a scatter plot, colored by the predict time, so
         that the color changes the further the forecast time is form the predict time.
         Defualt is False
    show_nsrdb : bool
         Show the line plot of the NSRDB modeleed value for the data type. Default
         is False
    scatter_measured : bool
         Show scatter plots for the measured (modeled and actual) values. Default
         isTrue.
    
    Results
    -------------------------------------------
    Plot of values from matplotlib package.
        
    '''
    if period:
        forecast_period = period
        #Begin rollingup forecast data by blocks basedon forecast_period
        if forecast_period > 24 or forecast_period < 1:
            raise ValueError("forecast_period must be an integer between 1 and 24 hours.")
    else:
        forecast_period = 1

    #select target fields:
    if data_type == 'temperature':
        forecast = 'temperature'
        actual = 'a_temperature'
        model = 'nsrdb_temp_air'
        hub = 'hub_air_temp'
        y_label = "Temperature (C)"
        if show_scatter== False:
            title = node_id + " Forecasted vs Observed and Modeled Temperature\nColored by Forecast Lead Time (" + str(forecast_period) + "H resampling)"
        else:
            title = node_id + " Forecasted vs Observed and Modeled Temperature"
        filename = node_id + "_forecast_actual_modeled_temp_" + str(forecast_period) + "h.png"
    elif  data_type == 'wind_speed':
        forecast = 'wind_speed' 
        actual = 'a_wind_speed'
        model = 'nsrdb_wind_speed'
        hub = 'hub_wind_speed'
        y_label = "Wind Speed (m/s)"
        if show_scatter== False:
            title = node_id + " Forecasted vs Observed and Modeled Wind Speed\nColored by Forecast Lead Time (" + str(forecast_period) + "H resampling)"
        else:
            title = node_id + " Forecasted vs Observed and Modeled Wind Speed"            
        filename = node_id + "_forecast_actual_modeled_ws_" + str(forecast_period) + "h.png"
    elif  data_type == 'clouds':
        print ('Hub has no different cloud comparison. Exiting')
        return
    else:
        print ('Unknown data type for comparison. Exiting')
        return

    #Forecast vs actual vas model
    #Prep and filter
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
    df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
    
    #Define date range of data to filter to
    start_date = pd.Timestamp(start_timestamp).tz_localize('UTC') 
    end_date = pd.Timestamp(end_timestamp).tz_localize('UTC') 
    df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]
    
    #For forecasts
    df_multi = df.set_index(['predict_time', 'forecast_time'])

    # Sort by forecast_time and predict_time so earliest predictions come first
    df = df.sort_values(by=['forecast_time', 'predict_time'])
    # Sort and prepare main plot data (first forecasts only)
    first_forecast_df = df.sort_values(by=['forecast_time', 'predict_time']) \
                          .drop_duplicates(subset='forecast_time', keep='first') \
                          .set_index('forecast_time') \
                          .sort_index()
    
    # Prepare scatter data
    scatter_df = df[['forecast_time', data_type, 'hour_diff']].dropna()
    
    scatter_df['forecast_time'] = pd.to_datetime(scatter_df['forecast_time'])
    
    #print('df size ' + str(df.shape))
    # Resample within each predict_time group
    df_resampled = (
        df_multi
        .groupby(level='predict_time')
        .resample(str(forecast_period) + 'h', level='forecast_time')  # resample forecast_time within each predict_time
        .agg({
            data_type : 'mean',       # or 'first', 'median', etc.
            'hour_diff': 'first',       # preserve the correct hour_diff for scatter color
        })
        .dropna()
        .reset_index()
    )
    # Calculate median forecast trend from resampled scatter data
    median_forecast = (
        df_resampled
        .groupby('forecast_time')[data_type]
        .median()
        .reset_index()
    )    
    #print('resampled size ' + str(df_resampled.shape))
    
    #Cretating boxplot dataset
    # Group wind_actual_resampled by forecast_time
    boxplot_data = (
        df_resampled
        .groupby('forecast_time')[data_type]
        .apply(list)  # create list of forecast values per time
    )    
        
    if forecast_period > 1:
        # Resample actual (NOAA) and modeled (NSRDB) using same forecast_period
        first_forecast_resampled = (
            first_forecast_df
            .resample(str(forecast_period) + 'h')  # resampling by forecast_period hours
            .agg({
                'predict_time': 'first',
                forecast: 'mean',
                actual: 'mean',   
                model: 'mean', 
                hub: 'mean'
            })
            .dropna()
            .reset_index()
        )    
    
    # Plot
    plt.figure(figsize=(16, 9), dpi=60)  
    
    #if wanting to do box and whiskers
    if box:
        # Create x positions for boxplots
        positions = mdates.date2num(boxplot_data.index)
        
        box_width_days = forecast_period / 24.0  # since 1 day = 24 hours
        
        # Slightly smaller to avoid overlap, e.g. 80% of width
        box_width_days *= 0.8    
    
        # Draw boxplots
        plt.boxplot(boxplot_data.values, positions=positions, widths=box_width_days, patch_artist=True,
                    boxprops=dict(facecolor='none', color='black', linewidth = 1.0),
                    medianprops=dict(color='black'))
    #End box plotting
    
    #Check for perodicity. If greater than 1, resample.
    if forecast_period > 1:
        if show_scatter:
            # Scatter: wind_speed colored by hour_diff
            sc = plt.scatter(
                df_resampled['forecast_time'],
                df_resampled[data_type],
                c=df_resampled['hour_diff'],
                cmap='cividis',
                alpha=0.7,
                label='Forecast'
            )
        # Line plot actual NOAA (resampled)
        plt.plot(first_forecast_resampled['forecast_time'], first_forecast_resampled[actual],
                 color='#00BCD4', label='NOAA (actual, ground)', linewidth=2)
        
        # Line plot: NSRDB Modeled (resampled)
        plt.plot(first_forecast_resampled['forecast_time'], first_forecast_resampled[model],
                 color='#D55E00', label='NSRDB (modeled)', linewidth=2)        
    
        # Line plot: Conus Modeled (resampled)
        plt.plot(first_forecast_resampled['forecast_time'], first_forecast_resampled[hub],
                 color='#800000', label='Conus (modeled, hub height)', linewidth=2)        

        # Median trend line for forecast values
        plt.plot(median_forecast['forecast_time'], first_forecast_resampled[forecast],
                 color='#CC79A7', linestyle='--', linewidth=2, label='Forecast Median')   

    else: # For standard file rate of hourly
        if show_scatter:
            # Scatter: wind_speed colored by hour_diff
                sc = plt.scatter(
                    scatter_df['forecast_time'],
                    scatter_df[data_type],
                    c=scatter_df['hour_diff'],
                    cmap='cividis',
                    alpha=0.7,
                    label='Forecast '
                )
            
        plt.plot(first_forecast_df.index, first_forecast_df[actual],
             color='#00BCD4', label='NOAA (actual, ground)', linewidth=2, )
        
        # Line plot:  NSRDB Modeled nsrdb_wind_speed
        if show_nsrdb:
            plt.plot(first_forecast_df.index, first_forecast_df[model],
                     color='#D55E00', label='NSRDB (modeled)', linewidth=2)

        # Line plot: Conus hub height wind_speed
        plt.plot(first_forecast_df.index, first_forecast_df[hub],
                 color='#800000', label='Conus(modeled, hub height)', linewidth=2)
    
        # Median trend line for forecast values
        plt.plot(median_forecast['forecast_time'], median_forecast[data_type],
                 color='#CC79A7', linestyle='--', linewidth=2, label='Forecast Median')   
               
    # Add colorbar
    if show_scatter:
        cbar = plt.colorbar(sc)
        cbar.set_label('Forecast Hour Difference')
    
    # Labels and legend
    plt.xlabel("Forecast Time",  fontsize=14)
    plt.ylabel(y_label,  fontsize=14)
    plt.title(title,  fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    #Adjust x axis lables if boxing.
    if box:
        # Format x-axis as datetime
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.xticks(rotation=45)
    
    if target_dir:
        plt.savefig(target_dir + filename)
    plt.show()
    
    if forecast_period > 1:
        return first_forecast_resampled
    else:
        return first_forecast_df

#-----------------------------------------------------------------------------------------------------
def compare_predict_time_forecasts_as_regression(target_dir, node_id, predict_date, df, data_type='temperature'):
    '''
    Called by the compare_predict_time_forecasts() method to do the regression scatter plot
    Creates a  scatter plot of changes in forecast based on the hour predictions are made 
    behaviors for a type of forecast (temp, wind, etc.). A plot is only one of the predict times in a
    predict day.
    
    Parameters
    -------------------------------------------
    node_id : str
         WECC Node id defined as partof REGROW Project.
    df : dataframe object
         Composite dataframe generated from forecast, actual and model data
    target_dir : str
         directory that the file is written to 
    data_type : str
          Which of the data types is the plotting to be done for. Possible values of
          temperature, wind_speed, and clouds
         
    Results
    -------------------------------------------
    Plot of values from matplotlib package.
        
    '''
    # Create a boolean mask of valid rows of timestamps. Then remove timezone
    mask = df['predict_time'].apply(lambda s: bool(sp.iso_space_offset_pattern.match(str(s))))
    df.loc[mask, "predict_time"] = df.loc[mask, "predict_time"].apply(sp.remove_offset_if_match)

    mask = df['forecast_time'].apply(lambda s: bool(sp.iso_space_offset_pattern.match(str(s))))
    df.loc[mask, "forecast_time"] = df.loc[mask, "forecast_time"].apply(sp.remove_offset_if_match)

    # Convert strings to datetime if not already
    df['predict_time'] = pd.to_datetime(df['predict_time'])
    df['forecast_time'] = pd.to_datetime(df['forecast_time'])   

    #Filter to one day
    # Filter rows where issue_time is exactly August 2 (any hour)
    df_predict = df[df["predict_time"].dt.date == datetime.fromisoformat(predict_date).date()]
    df_predict['hour_predict'] = df_predict['predict_time'].dt.hour

    #Loop and do a plot for each predict time in the predict day
    predict_times = [5, 11, 17, 23]
    for target_predict_time in predict_times:
        # Create the plot
        plt.figure(figsize=(10, 6))
    
        if data_type == 'temperature':
            x_data="temperature"
            y_data="a_temperature"
            title_tag = 'Temperature'
            # Customize the plot
            plt.xlabel("Forecast Temperature (C)")
            plt.ylabel("Actual Temperature (C)")
        elif data_type == 'clouds':
            x_data="clouds"
            y_data="a_cloud_cover"
            title_tag = 'Cloud Cover'
            # Customize the plot
            plt.xlabel("Forecast Cloud Cover (%)")
            plt.ylabel("Actual Temperature (%)")
        elif data_type == 'wind_speed':
            x_data="wind_speed"
            y_data="a_wind_speed"
            title_tag = 'Wind Speed'
            # Customize the plot
            plt.xlabel("Forecast Wind Speed (m/s)")
            plt.ylabel("Actual Wind Speed (m/s)")
        else:
            print (f'Unknown data type {data_type} passed in. Exiting...')
            return
    
        if target_predict_time == 5:
            file_tag = '0500'
            df_filtered = df_predict[df_predict["predict_time"].dt.time == datetime.strptime("05:00:00", "%H:%M:%S").time()]
            plt.title("Forecast vs Actual " + title_tag + " (Predict Time: 05:00)")
        elif target_predict_time == 11:
            file_tag = '1100'
            plt.title("Forecast vs Actual " + title_tag + "  (Predict Time: 11:00)")
            df_filtered = df_predict[df_predict["predict_time"].dt.time == datetime.strptime("11:00:00", "%H:%M:%S").time()]
        elif target_predict_time == 17:
            file_tag = '1700'
            df_filtered = df_predict[df_predict["predict_time"].dt.time == datetime.strptime("17:00:00", "%H:%M:%S").time()]
            plt.title("Forecast vs Actual " + title_tag + "  (Predict Time: 17:00)")
        elif target_predict_time == 23:
            file_tag = '2300'
            df_filtered = df_predict[df_predict["predict_time"].dt.time == datetime.strptime("23:00:00", "%H:%M:%S").time()]
            plt.title("Forecast vs Actual " + title_tag + "  (Predict Time: 23:00)")
        
        sns.scatterplot(
            x=x_data,
            y=y_data,
            style="forecast_day",
            data=df_filtered,
            palette="Blues",
            s=30,
        )
        
        sns.regplot(
            x=x_data,
            y=y_data,
            data=df_filtered,
            scatter=False,  # Don't replot scatter points
            color="grey",
            #ci=None,  # Remove confidence intervals
        )
    
        # Customize the legend to only show Forecast Day
        handles, labels = plt.gca().get_legend_handles_labels()
        legend_labels = [label for label in labels if label.isdigit()]  # Keep only numeric labels (Forecast Days)
        plt.legend(handles=handles[1:], labels=legend_labels, title="forecast_day", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        #save
        plt.savefig("C:/ReGROW/target_nodes_forecast_to_actual/plots/" + node_id + "_" + data_type + "_regress_"+ predict_date + "_ " + file_tag + ".png", dpi=200)
        plt.show()
    
    return

#-----------------------------------------------------------------------------------------------------
def compare_predict_time_forecasts(node_id, df, predict_date, data_type='temperature', heat_map=True, regres=False, target_dir = ''):
    '''
    Creates a mixed scatter and line plot of changes in forecast based on the date predictions are made 
   based on a type of forecast (temp, wind, etc.)
    
    Parameters
    -------------------------------------------
    node_id : str
         WECC Node id defined as partof REGROW Project.
    df : dataframe object
         Composite dataframe generated from forecast, actual and model data
    predict_date : str
          iso 8601 date that evalaution is made from
    target_dir : str
         directory that the file is written to
         Default is empty (no write)
    data_type : str
          Which of the data types is the plotting to be done for. Possible values of
          temperature, wind_speed, and clouds.
          Default = temperature
         
    Results
    -------------------------------------------
    Plot of values from matplotlib package.
        
    '''
    #Perform scatter plot
    if regres:
        compare_predict_time_forecasts_as_regression(target_dir, node_id, predict_date, df, data_type)
        return
    
    #Perfom line plot and heat maps
    # A simple dict that maps the hour to a chosen color
    color_map = {
        5:  'red',     # For 05:00
        11: 'blue',    # For 11:00
        17: 'green',   # For 17:00
        23: 'purple'   # For 23:00
    }
    
    # Also define a friendly label for the legend
    label_map = {
        5:  '05:00 Issue',
        11: '11:00 Issue',
        17: '17:00 Issue',
        23: '23:00 Issue'
    }
    
    
    # Create a boolean mask of valid rows of timestamps. Then remove timezone
    mask = df['predict_time'].apply(lambda s: bool(sp.iso_space_offset_pattern.match(str(s))))
    df.loc[mask, "predict_time"] = df.loc[mask, "predict_time"].apply(sp.remove_offset_if_match)

    mask = df['forecast_time'].apply(lambda s: bool(sp.iso_space_offset_pattern.match(str(s))))
    df.loc[mask, "forecast_time"] = df.loc[mask, "forecast_time"].apply(sp.remove_offset_if_match)

    # Convert strings to datetime if not already
    df['predict_time'] = pd.to_datetime(df['predict_time'])
    df['forecast_time'] = pd.to_datetime(df['forecast_time'])   

    #Calculate differences between forecast and actual values
    df['residual_temp'] = df['temperature'] - df['a_temperature']
    df['residual_cloud_cover'] = df['clouds'] - df['a_cloud_cover']
    df['residual_wind_speed'] = df['wind_speed'] - df['a_wind_speed']
    df['residual_wind_dir'] = df['wind_deg'] - df['a_wind_dir']
    
    #Filter to one day
    # Filter rows where issue_time is exactly August 2 (any hour)
    df_predict = df[df["predict_time"].dt.date == datetime.fromisoformat(predict_date).date()]
    df_predict['hour_predict'] = df_predict['predict_time'].dt.hour
    
    #Set up pivoted frame based on data type
    if data_type == 'temperature':
        temp_pivot = df_predict.pivot_table(
            index='forecast_time',
            columns='hour_predict',
            values=['temperature', 'a_temperature']
        )
    elif data_type == 'wind_speed':
        temp_pivot = df_predict.pivot_table(
            index='forecast_time',
            columns='hour_predict',
            values=['wind_speed', 'a_wind_speed']
        )
        
    elif data_type == 'clouds':
        temp_pivot = df_predict.pivot_table(
            index='forecast_time',
            columns='hour_predict',
            values=['clouds', 'a_cloud_cover']
        )
    else:
        print (f'Unknown data type {data_type} passed in. Exiting...')
        return
        
    fig, ax = plt.subplots(figsize=(12, 6))
    already_labeled_hours = set()  # Keep track of which hours are already in the legend
    for col in temp_pivot.columns:
        #Skip actual temps from scatter plot
        #if 'a_temperature' in col:
        #if 'a_cloud_cover' in col:
        if 'a_wind_speed' in col:
            continue
        #get predict  hour
        hour = col[1]
        # Choose color based on hour, default to gray or black if somehow not in color_map
        c = color_map.get(hour, 'black')
       
        # Decide the label: only label once per hour
        if hour not in already_labeled_hours:
            label = label_map.get(hour, f"{hour}:00 Issue")  # fallback label
            already_labeled_hours.add(hour)
        else:
            label = None
       
        # Plot scatter for this column
        ax.scatter(
            temp_pivot.index,   # X values = forecast_time
            temp_pivot[col],    # Y values = the temperature data
            color=c,
            label=label,        # Only label if it's the first time we've seen this hour
            alpha=0.7
        )
     
    # Plot actual temperatures as a line
    if data_type == 'temperature':
        ax.set_title('Predictions from ' + predict_date + ' Temperature vs Forecast Time for Each Issue Time')
        ax.plot(temp_pivot.index, temp_pivot[('a_temperature', 5)], label='Actual Temperature', color='black')
        ax.set_ylabel('Temperature (°C)')
    elif data_type == 'wind_speed':
        ax.set_title('Predictions from ' + predict_date + '  Cloud Cover vs Forecast Time for Each Issue Time')
        ax.plot(temp_pivot.index, temp_pivot[('a_wind_speed', 5)], label='Actual Wind Speed', color='black')
        ax.set_ylabel('Wind Speed (m/s)')
    elif  data_type == 'clouds':
        ax.set_title('Predictions from ' + predict_date + '  Wind Speed vs Forecast Time for Each Issue Time')
        ax.plot(temp_pivot.index, temp_pivot[('a_cloud_cover', 5)], label='Actual Cloud Cover', color='black')
        ax.set_ylabel('Cloud Cover (%)')
       
    ax.set_xlabel('Forecast Time')
    plt.legend(title='Issue Time', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust space on the right side so the legend fits
    plt.subplots_adjust(right=0.8)  # Lower numbers = more space for legend
    plt.show()
    
    # Heat map of residuals
    if heat_map:
        #Set up pivoted frame based on data type
        if data_type == 'temperature':
            res_map = df_predict.pivot_table(
                index='hour_predict',
                columns='forecast_time',
                values='residual_temp',
                aggfunc='mean'  # or np.mean, or any other if duplicates exist
            )
        elif data_type == 'wind_speed':
            res_map = df_predict.pivot_table(
                index='hour_predict',
                columns='forecast_time',
                values='residual_wind_speed', 
                aggfunc='mean'  # or np.mean, or any other if duplicates exist
            )
        elif data_type == 'clouds':
            res_map = df_predict.pivot_table(
                index='hour_predict',
                columns='forecast_time',
                values='residual_cloud_cover', 
                aggfunc='mean'  # or np.mean, or any other if duplicates exist
            )
        plt.figure(figsize=(10, 6))  # adjust as needed
        ax = sns.heatmap(
            #res_map, 
            res_map.abs(), 
            cmap='RdBu_r',    # Red-Blue reversed: negative=blue, positive=red
            center=0,         # Make 0 white (the midpoint of the colormap)
            #annot=True,       # optional: show numeric values in each cell
            #fmt=".1f"         # decimal format for annotations
        )
        
        if data_type == 'temperature':
            plt.title(node_id + ' - Residuals of  Predictions from ' + predict_date + '  Temperature Predictions by Hour and Prediction Time')
        elif data_type == 'wind_speed':
            plt.title(node_id + ' - Residuals of Predictions from ' + predict_date + '  Wind Speed Predictions by Hour and Prediction Time')
        elif data_type == 'clouds':
            plt.title(node_id + ' - Residuals of Predictions from ' + predict_date + '  Cloud Cover Predictions by Hour and Prediction Time')
        plt.xlabel('Hour Diff (hours after forecast)')
        plt.ylabel('Predict Time (hour of day)')
        
        all_labels = res_map.columns.strftime('%Y-%m-%d %H:%M')
        positions = np.arange(1, len(all_labels), 5)
        selected_labels = [all_labels[i] for i in positions]
        ax.set_xticks(positions)
        ax.set_xticklabels(selected_labels, rotation=45, ha='right')
        plt.tight_layout()
        
        #save
        if target_dir :
            plt.savefig(target_dir + node_id + "_wspeed_map_forecast_vs_actual_abs_20200802-2020808.png", dpi=200)
       #set_title
        plt.show()       
    return

#-----------------------------------------------------------------------------------------------------
def explore_sample_forecast_clouds_vs_actual(forecast_data, actual_data):
    '''
    Examines in comparison the forecast values to actual values for a set of data files
    
    Parameters
    ---------------------------
    forecast_data : json
         A json file of forecast data
    actual_data : dataframe obj
        A pandas data frame containg the values extracted from a targeted CSV
        
    Results
    --------------------------
    A plot of the daily values in comparison.
    '''
    sky_cover = [entry["value"] for entry in forecast_data["sky_cover"]["values"]]
    timestamps = [entry["validTime"] for entry in forecast_data["sky_cover"]["values"]]
    # Optional reformat
    formatted_timestamps = [convert_timestamp(ts) for ts in timestamps]
    df_forecast = pd.DataFrame({'time': formatted_timestamps, 'forecast_cloud_cover' : sky_cover})

    # Parse the actual data
    # Assuming the CSV file has columns like 'timestamp', 'min_temperature', 'max_temperature'
    actual_data['time'] = pd.to_datetime(actual_data['time'])  # Convert to datetime if needed
    actual_data['time'] = actual_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_weather = pd.merge(actual_data, df_forecast, on='time', how='left')

    # Plotting the wind speed and using gust speed to color the points
    plt.figure(figsize=(12, 6))
    #plot forecast cloud cover
    plt.plot(df_weather['time'], df_weather['forecast_cloud_cover'],
             color='blue',
             marker ='o',
             linestyle='-',
             label='Forecast Cloud Cover (%)')
    # Plot actual cloud cover
    plt.plot(df_weather['time'],
             df_weather['cloud_cover'],
             color='darkred',
             marker='*',
             linestyle='-',
             label='Actual Cloud Cover (%)')
    
    plt.xticks(df_weather['time'][::10], rotation=45, ha='right')
    plt.xlabel('Timestamp')
    plt.ylabel('Cloud Cover(%)')
    plt.title('Node 9q603v : Actual vs. Forecasted Cloud Coverage by Time')
    plt.legend()
    plt.tight_layout()
    plt.show()        
    return


#-----------------------------------------------------------------------------------------------------
def explore_sample_forecast_clouds_vs_actual_combined_df(df):
    '''
    Examines in comparison the forecast values to actual values for a node
    
    Parameters
    ---------------------------
    df : dataframe object
         a combined dataframe of actual and forecast weather 
        
    Results
    --------------------------
    A plot of the daily values in comparison.
    '''
    node_id =df["node_id"].iloc[0]

    # Plotting the wind speed and using gust speed to color the points
    plt.figure(figsize=(12, 6))
    #plot forecast cloud cover
    plt.plot(df['forecast_time'], df['clouds'],
             color='blue',
             marker ='o',
             linestyle='-',
             label='Forecast Cloud Cover (%)')
    # Plot actual cloud cover
    plt.plot(df['forecast_time'],
             df['a_cloud_cover'],
             color='darkred',
             marker='*',
             linestyle='-',
             label='Actual Cloud Cover (%)')
    
    plt.xticks(df['forecast_time'][::10], rotation=45, ha='right')
    plt.xlabel('Timestamp')
    plt.ylabel('Cloud Cover(%)')
    plt.title(node_id + ' - Actual vs. Forecasted Cloud Coverage by Time')
    #plt.legend()
    plt.tight_layout()
    plt.show()        
    return


#-----------------------------------------------------------------------------------------------------
def explore_forecast_clouds_stacked (df):
    '''
    
    
    '''
    node_id =df["node_id"].iloc[0]
    
    # Convert 'predict_time' to datetime if not already
    df["predict_time"] = pd.to_datetime(df["predict_time"])
    
    # Define date range for filtering (convert to datetime)
    start_date = pd.to_datetime("2018-01-01 00:00:00-07:00")
    end_date = pd.to_datetime("2019-12-31 23:59:59-07:00")
    
    # Apply filtering
    df_filtered = df[(df["day_diff"] == 0) & (df["predict_time"].between(start_date, end_date))]

    # Filter data where day_diff == 0
    #df_filtered = df[df["day_diff"] == 0]
    
    # Plot settings
    plt.figure(figsize=(10, 6))
    
    # Loop through each unique predict_time and plot separately
    for predict_time, group in df_filtered.groupby("predict_time"):
        # Sort for proper plotting
        group = group.sort_values(by="hour_diff")  
        
        # Generate more points for a smooth curve
        x_smooth = np.linspace(group["hour_diff"].min(), group["hour_diff"].max(), 200)
        
        # Apply linear interpolation
        linear_interp = interp1d(group["hour_diff"], group["clouds"], kind='linear', fill_value="extrapolate")
        y_smooth = linear_interp(x_smooth)

        # Plot smoothed curve
        plt.plot(x_smooth, y_smooth, linestyle='-', color='blue', linewidth=0.8)
        #plt.plot(group["hour_diff"], group["clouds"], linestyle='-', linewidth=0.8, color='blue')
    
    # Labels and title
    plt.xlabel("Hour Difference")
    plt.ylabel("Cloud Cover (%)")
    plt.title( node_id + " - Cloud Cover (%) vs Hour Difference for Each Predcit Time (2018-2019)")
    plt.grid(True)
    
    # Show plot
    plt.show()
    
    
    ## Count the number of forecasts for each hour_diff
    #forecast_counts = df_filtered.groupby('hour_diff').size().reset_index(name='forecast_count')
    
    ## Merge with cloud coverage data
    #df_merged = df_filtered.groupby('hour_diff')['clouds'].mean().reset_index()
    
    ## Merge the two DataFrames on 'hour_diff'
    #df_plot = pd.merge(forecast_counts, df_merged, on='hour_diff')
    
    ## Create a pivot table with hour_diff as rows, forecast_count as columns, and cloud coverage as values
    #heatmap_data = df_plot.pivot_table(index='hour_diff', columns='forecast_count', values='clouds', aggfunc='mean')
    
    ## Plot heatmap
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Cloud Cover (%)'})
    
    ## Labels and title
    #plt.xlabel('Number of Forecasts')
    #plt.ylabel('Hour Difference')
    #plt.title('Heatmap of Hour Difference vs Number of Forecasts with Cloud Cover as Z Dimension')
    
    ## Show plot
    #plt.show()
    
    
#-----------------------------------------------------------------------------------------------------
def explore_sample_forecast_winds(forecast_data):
    '''
    Looking at the daily winds for a given forecast data file
    
    Parameters
    ---------------------------
    forecast_data : json
         A json file of forecast data
        
    Results
    --------------------------
    A plot of the daily values from file.
    '''
    # Extract wind speed, gust values, and timestamps from each entry in the "list" key
    wind_speed = [entry["wind"]["speed"] for entry in forecast_data["list"]]
    gust_speed = [entry["wind"]["gust"] for entry in forecast_data["list"]]
    timestamps = [entry["dt_txt"] for entry in forecast_data["list"]]
    
    # Plotting the wind speed and using gust speed to color the points
    plt.figure(figsize=(12, 6))
    # First plot the line (so it appears behind the scatter points)
    plt.plot(timestamps, wind_speed, color='gray', label='Wind Line')
    scatter = plt.scatter(timestamps, wind_speed, c=gust_speed, cmap='viridis', s=80, edgecolor='black')
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Timestamp')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed vs. Time, Color by Gust Intensity - Jan 8, 2025')
    plt.colorbar(scatter, label='Gust Speed (m/s)')
    plt.tight_layout()
    plt.show()        
    return
    
#-----------------------------------------------------------------------------------------------------
def explore_sample_forecast_temperatures(forecast_data):
    '''
    Looking at the daily temps for a given forecast data file
    
    Parameters
    ---------------------------
    forecast_data : json
         A json file of forecast data
        
    Results
    --------------------------
    A plot of the daily values from file.
    '''
    # Extract temperature and humidity, and timestamps from each entry in the "list" key
    kelvin_temps = [entry["main"]["temp"] for entry in forecast_data["list"]]
    celsius_temps = [temp - 273.15 for temp in kelvin_temps]
    humidity = [entry["main"]["humidity"] for entry in forecast_data["list"]]
    timestamps = [entry["dt_txt"] for entry in forecast_data["list"]]
    
    # Plotting the temps and using humidity to color the points
    plt.figure(figsize=(12, 6))
    # First plot the line (so it appears behind the scatter points)
    plt.plot(timestamps, celsius_temps, color='gray', label='Temperature Line')
    scatter = plt.scatter(timestamps, celsius_temps, c=humidity, cmap='RdBu', s=80, edgecolor='black')
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature vs. Time, Color by Humidity - Jan 8, 2025')
    plt.colorbar(scatter, label='Humidity (%)')
    plt.tight_layout()
    plt.show()        
    return

#-----------------------------------------------------------------------------------------------------
def explore_sample_forecast_clouds(forecast_data):
    '''
    Looking at the daily cloud coverage for a given forecast data file
    
    Parameters
    ---------------------------
    forecast_data : json
         A json file of forecast data
        
    Results
    --------------------------
    A plot of the daily values from file.
    '''
    # Extract cloud coverage and timestamps from each entry in the "list" key
    cloud_cover = [entry["clouds"]["all"] for entry in forecast_data["list"]]
    timestamps = [entry["dt_txt"] for entry in forecast_data["list"]]
    
    # Plotting the cloud coverage
    plt.figure(figsize=(12, 6))
    # First plot the line (so it appears behind the scatter points)
    plt.plot(timestamps, cloud_cover, color='gray', label='cloud Line')
    plt.scatter(timestamps, cloud_cover, color='blue', s=80, edgecolor='black')
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Timestamp')
    plt.ylabel('Cloud Cover (%)')
    plt.title('Cloud coverage vs. Time - Jan 8, 2025')
    plt.tight_layout()
    plt.show()        
    return

#-----------------------------------------------------------------------------------------------------
def explore_sample_forecast_temps_vs_actual(forecast_data, actual_data, nws_flag):
    '''
    Compare values of predicted min and max temps to actual values
    
    Parameters
    ---------------------------
    forecast_data : json
         A json file of forecast data
    actual_data : dataframe obj
        A pandas data frame containg the values extracted from a targeted CSV
    nws_flag : bool
        Passed in from arg parameters. Is the comparison source nws forecasts or Open weather
        
    Results
    --------------------------
    A plot of the daily values in comparison.
    '''
    if nws_flag:
        # Extract dates and values for max and min temperatures
        forecast_max_times  = [datetime.fromisoformat(entry["validTime"].split("/")[0]) for entry in forecast_data["max_temperature"]["values"]]
        forecast_max_values  = [entry["value"] for entry in forecast_data["max_temperature"]["values"]]
        forecast_min_times  = [datetime.fromisoformat(entry["validTime"].split("/")[0]) for entry in forecast_data["min_temperature"]["values"]]
        forecast_min_values  = [entry["value"] for entry in forecast_data["min_temperature"]["values"]]           
        
        # Parse the actual data
        # Assuming the CSV file has columns like 'timestamp', 'min_temperature', 'max_temperature'
        actual_data['time'] = pd.to_datetime(actual_data['time'])  # Convert to datetime if needed
        
        plt.figure(figsize=(12, 6))
        # Forecast Max and Min temperature scatter plot
        plt.scatter(forecast_max_times, forecast_max_values, color='red', label='Forecast Max Temperature (C)')
        plt.scatter(forecast_min_times, forecast_min_values, color='blue', label='Forecast Min Temperature (C)')
        # Plot actual temperatures
        plt.plot(actual_data['time'],
                 actual_data['temperature_2m'],
                 color='darkred',
                 marker='*',
                 linestyle='-',
                 label='Actual Temperature (C)'
                 )
         
        plt.xlabel('Date')
        plt.title('Node 9q603v : NWS - Actual vs. Forecasted Min and Max Temperatures')
        plt.xticks(rotation=45)
                   
        
    else: #open Weather       
        # Extract temp_min, temp_max, and timestamps from each entry in the "list" key
        temp_min_values = [entry["main"]["temp_min"] for entry in forecast_data["list"]]
        temp_max_values = [entry["main"]["temp_max"] for entry in forecast_data["list"]]
        #timestamps = [entry["dt_txt"] for entry in forecast_data["list"]]
        timestamps_utc = [entry["dt_txt"] for entry in forecast_data["list"]]
        timestamps_utc = pd.to_datetime(timestamps_utc).tz_localize('UTC')
        timestamps_la = timestamps_utc.tz_convert('America/Los_Angeles')
        timestamps_la_iso = timestamps_la.strftime('%Y-%m-%d %H:%M:%S')
        timestamps = timestamps_la_iso.tolist()
        # Optional: Convert temperatures from Kelvin to Celsius
        temp_min_celsius = [temp - 273.15 for temp in temp_min_values]
        temp_max_celsius = [temp - 273.15 for temp in temp_max_values]    # Plotting the temperature range for each timestamp as a vertical line from temp_min to temp_max in Celsius        
        
        df_ow_forecast = pd.DataFrame({ 'time' : timestamps, 'temp_min' : temp_min_celsius, 'temp_max' : temp_max_celsius})

         # Parse the actual data
        # Assuming the CSV file has columns like 'timestamp', 'min_temperature', 'max_temperature'
        actual_data['time'] = pd.to_datetime(actual_data['time'])  # Convert to datetime if needed
        #actual_data['time'] = actual_data['time'].dt.tz_localize(UTC)
        actual_data['time'] = actual_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df_weather = pd.merge(actual_data, df_ow_forecast, on='time', how='left')

        plt.figure(figsize=(12, 6))
        #plt.vlines(timestamps, temp_min_celsius, temp_max_celsius, color='b', alpha=0.6, linewidth=2)
        plt.scatter(df_weather['time'], df_weather['temp_max' ], color='red', edgecolor='black', label='Forecast Max Temperature (°C)', s=80, zorder=3)
        plt.scatter(df_weather['time'], df_weather['temp_min' ], color='blue', edgecolor='black', label='Forecast Min Temperature (°C)', s=20, zorder=3)

        # Plot actual temperatures
        plt.plot(df_weather['time'],
                 df_weather['temperature_2m'],
                 color='darkred',
                 marker='*',
                 linestyle='-',
                 label='Actual Temperature (C)'
                 )

        plt.xlabel('Timestamp')
        plt.xticks(df_weather['time'][::10], rotation=45, ha='right')
        plt.title('Node 9q603v : OpenWeather - Forecasted Min and Max Temperatures)')

    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


#-----------------------------------------------------------------------------------------------------
def explore_accuracy_forecast_temp_vs_actual(df, compute_residuals):
    '''
    Compare values of predicted min and max temps to actual values
    
    Parameters
    ---------------------------
    df : dataframe
         Composite data frame of actual and forecasts
    node : str
        target node.
    compute_residuals : bool
         If not previously computed, executer then write out as file.
        
    Results
    --------------------------
    A plot of the daily values in comparison.
    '''
    choice = 'density_heat'
       
    node_id =df["node_id"].iloc[0]
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
    df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
    start_date = pd.Timestamp("2020-07-25 00:00:00").tz_localize('UTC') 
    end_date = pd.Timestamp("2020-08-15 23:59:59").tz_localize('UTC') 
    #Full file
    #start_date = pd.Timestamp("2018-01-01 00:00:00").tz_localize('UTC') 
    #end_date = pd.Timestamp("2022-12-31 23:59:59").tz_localize('UTC') 
    
    ## Filter for forecast_time between start_date and end_date
    df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]
    # Compute residuals, otherwise df has residulas in it already
    if compute_residuals:
        df["residual"] = df["temperature"] - df["a_temperature"]
    
        df.to_csv('C:/ReGROW/target_nodes_forecast_to_actual/with_residuals./' + node_id + '_forecast_vs_actrual_with_residual_temps.csv', index=False)


    if  choice == 'scatter': 
        # Scatter plot of residuals against hour_diff
        plt.figure(figsize=(10, 6))
        plt.scatter(df["hour_diff"], df["residual"], color="blue", alpha=0.5)
        
        plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Reference line
        plt.xlabel("Hour Difference")
        plt.ylabel("Residual (temp - a_temp)")
        plt.title("Forecast Accuracy Over Time")
        plt.grid(True)
        plt.show()
        
    if choice == 'binned_scatter' :
        # Bin data into groups
        df['hour_diff_bin'] = pd.cut(df['hour_diff'], bins=np.arange(0, df['hour_diff'].max(), 6))  # Adjust bin size
        
        # Compute mean and std for each bin
        binned_stats = df.groupby('hour_diff_bin')['residual'].agg(['mean', 'std'])
        
        # Plot with error bars
        plt.figure(figsize=(8,5))
        plt.errorbar(binned_stats.index.categories.mid, binned_stats['mean'], yerr=binned_stats['std'], fmt='o-', capsize=5)
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual (Predicted - Actual Temperature)')
        plt.title('Mean Residuals & Variability Over Forecast Lead Time')
        plt.show()
#        choice =   'density_heat'
        
    if choice == 'density_heat':
        #plt.figure(figsize=(8,3))
        g = sns.JointGrid(data=df, x="hour_diff", y="residual", marginal_ticks=True)
        
        # Density Plot
        g.plot_joint(sns.kdeplot, fill=True,  cmap='PuOr')
        
        # Marginal Distributions (Histograms or KDEs)
        g.plot_marginals(sns.histplot, kde=True, bins=20)                               
        
        # Remove X-axis marginal to keep only Y
        g.ax_marg_x.set_visible(False)

        g.figure.suptitle("Density Plot of Forecast Lead Time vs. Temperature Residuals", fontsize=14)
        g.ax_joint.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
        g.ax_joint.set_ylabel("Residuals (Temperature - Actual Temperature)", fontsize=12)

        plt.show()
#      choice =   'heat_vnm'
        
    if choice == 'rolling_std':
        df_sorted = df.sort_values('hour_diff')
        df['rolling_std'] = df_sorted['residual'].rolling(window=10, min_periods=5).std()
        
        plt.figure(figsize=(8,5))
        plt.plot(df_sorted['hour_diff'], df['rolling_std'], color='red', label='Rolling Std Dev')
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual Variability in Temps')
        plt.title('Expanding Error Over Forecast Lead Time')
        plt.legend()
        plt.show()       
#        choice =   'heat_vnm'
        
    if choice == 'heat_vnm':
        # Bin the data
        df['hour_bin'] = pd.cut(df['hour_diff'], bins=np.arange(0, df['hour_diff'].max() + 6, 6))  # 6-hour bins
        df['residual_bin'] = pd.cut(df['residual'], bins=np.linspace(df['residual'].min(), df['residual'].max(), 20))  # Residual bins
        
        # Pivot table to compute mean residuals
        heatmap_data = df.pivot_table(index='residual_bin', columns='hour_bin', values='residual', aggfunc='mean')
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='PuOr', center=0, annot=False, cbar=True)
        
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual Bins')
        plt.title('Heatmap of Mean Temperature Residuals Over Forecast Lead Time')
        plt.show()
        
    return


#-----------------------------------------------------------------------------------------------------
def explore_accuracy_forecast_windspeed_vs_actual(df, compute_residuals):
    '''
    Compare values of predicted min and max temps to actual values
    
    Parameters
    ---------------------------
    df : dataframe
         Composite data frame of actual and forecasts
    node : str
        target node.
    compute_residuals : bool
         If not previously computed, executer then write out as file.
        
    Results
    --------------------------
    A plot of the daily values in comparison.
    '''
    choice = 'density_heat'
    
    
    node_id =df["node_id"].iloc[0]
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
    df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
    start_date = pd.Timestamp("2020-07-25 00:00:00").tz_localize('UTC') 
    end_date = pd.Timestamp("2020-08-15 23:59:59").tz_localize('UTC') 
    #Full file
    #start_date = pd.Timestamp("2018-01-01 00:00:00").tz_localize('UTC') 
    #end_date = pd.Timestamp("2022-12-31 23:59:59").tz_localize('UTC') 
    
    ## Filter for forecast_time between start_date and end_date
    df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]

    # Compute residuals, otherwise df has residulas in it already
    if compute_residuals:
        df["residual"] = df["wind_speed"] - df["a_wind_speed"]
        df.to_csv('C:/ReGROW/target_nodes_forecast_to_actual/with_residuals./' + node_id + '_forecast_vs_actual_with_residual_ws.csv', index=False)
    
    if  choice == 'scatter':        
        # Scatter plot of residuals against hour_diff
        plt.figure(figsize=(10, 6))
        plt.scatter(df["hour_diff"], df["residual"], color="blue", alpha=0.5)
        
        plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Reference line
        plt.xlabel("Hour Difference")
        plt.ylabel("Residual (wind speed - a-wind speed)")
        plt.title("Forecast Accuracy Over Time")
        plt.grid(True)
        plt.show()
        
    if choice == 'binned_scatter' :
        # Bin data into groups
        df['hour_diff_bin'] = pd.cut(df['hour_diff'], bins=np.arange(0, df['hour_diff'].max(), 6))  # Adjust bin size
        
        # Compute mean and std for each bin
        binned_stats = df.groupby('hour_diff_bin')['residual'].agg(['mean', 'std'])
        
        # Plot with error bars
        plt.figure(figsize=(8,5))
        plt.errorbar(binned_stats.index.categories.mid, binned_stats['mean'], yerr=binned_stats['std'], fmt='o-', capsize=5)
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual (Predicted - Actual Wind Speed)')
        plt.title('Mean Residuals & Variability Over Forecast Lead Time')
        plt.show()
#        choice =   'density_heat'
        
    if choice == 'density_heat':
        plt.figure(figsize=(8,5))
 
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Figures of merit      
        # Assuming you have 'forecast' and 'actual' columns in your dataframe:
        df['absolute_error'] = abs(df['residual'])
        
        # Calculate Mean Absolute Error (MAE)
        MAE = df['absolute_error'].mean()
        print(f"Mean Absolute Error  for Temeprature (MAE): {MAE}")
        
        # Calculate Root Mean Squared Error (RMSE)
        RMSE = ( df["residual"] ** 2).mean() ** 0.5
        print(f"Root Mean Squared Error (RMSE): {RMSE}")
        
        g = sns.JointGrid(data=df, x="hour_diff", y="residual", marginal_ticks=True)
        
        # Density Plot
        g.plot_joint(sns.kdeplot, fill=True,  cmap='PuOr')
        
        # Marginal Distributions (Histograms or KDEs)
        g.plot_marginals(sns.histplot, kde=True, bins=20)                               
        
        # Remove X-axis marginal to keep only Y
        g.ax_marg_x.set_visible(False)

        g.figure.suptitle("Density Plot of Forecast Lead Time vs. Wind Speed Residuals", fontsize=14)
        g.ax_joint.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
        g.ax_joint.set_ylabel("Residuals (Wind Speed - Actual Wind Speed)", fontsize=12)

        plt.show()
#      choice =   'heat_vnm'
        
    if choice == 'rolling_std':
        df_sorted = df.sort_values('hour_diff')
        df['rolling_std'] = df_sorted['residual'].rolling(window=10, min_periods=5).std()
        
        plt.figure(figsize=(8,5))
        plt.plot(df_sorted['hour_diff'], df['rolling_std'], color='red', label='Rolling Std Dev')
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual Variability in wind speed')
        plt.title('Expanding Error Over Forecast Lead Time')
        plt.legend()
        plt.show()       
#        choice =   'heat_vnm'
        
    if choice == 'heat_vnm':
        # Bin the data
        df['hour_bin'] = pd.cut(df['hour_diff'], bins=np.arange(0, df['hour_diff'].max() + 6, 6))  # 6-hour bins
        df['residual_bin'] = pd.cut(df['residual'], bins=np.linspace(df['residual'].min(), df['residual'].max(), 20))  # Residual bins
        
        # Pivot table to compute mean residuals
        heatmap_data = df.pivot_table(index='residual_bin', columns='hour_bin', values='residual', aggfunc='mean')
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='PuOr', center=0, annot=False, cbar=True)
        
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual Bins')
        plt.title('Heatmap of Mean wind speed Residuals Over Forecast Lead Time')
        plt.show()
        
    return


#-----------------------------------------------------------------------------------------------------
def explore_accuracy_forecast_clouds_vs_actual(df,
                                               start_timestamp,
                                               end_timestamp, 
                                               plot_type='scatter',
                                               cascade=False,
                                               density_merits=False, 
                                               compute_residuals = False):
    '''
    Compare values of predicted min and max temps to actual values
    
    Parameters
    ---------------------------
    df : dataframe
         Composite data frame of actual and forecasts
    node : str
        target node.
    start_date : str
         ISO 8601 date to define start period of prediction days in forecast data
    end_date : str
         ISO 8601 date to define the last day of the period of prediction days in forecast data
    plot_type : str
         Type of plot to build for this exploration. Default is scatter
         Other types: binned_scatter
                                  density_heat         
                                  rolling_std
                                  heat_vnm
    cascade : bool
          Defines wheter to walk through and do each plot type as an
          exploration for best visualization. Leave plot type at default
          and it will start
    density_merits : bool
         calculate MAE and RSME for density plot
      compute_residuals : bool
         If not previously computed, executer then write out as file.
       
    Results
    --------------------------
    A plot of the daily values in comparison.
    '''
    node_id =df["node_id"].iloc[0]
    df['forecast_time'] = pd.to_datetime(df['forecast_time'], utc=True)
    df['predict_time'] = pd.to_datetime(df['predict_time'], utc=True)
    #Two weeks centered on event
    start_date = pd.Timestamp(start_timestamp).tz_localize('UTC') 
    end_date = pd.Timestamp(end_timestamp).tz_localize('UTC') 
    
    ## Filter for forecast_time between start_date and end_date
    df = df[(df["forecast_time"] >= start_date) & (df["forecast_time"] <= end_date)]

    # Compute residuals, otherwise df has residulas in it already
    if compute_residuals:
        df["residual"] = df["clouds"] - df["a_cloud_cover"]
        df.to_csv('C:/ReGROW/target_nodes_forecast_to_actual/with_residuals./' + node_id + '_forecast_vs_actrual_with_residual_clouds.csv', index=False)
    
    if  plot_type == 'scatter':        
        # Scatter plot of residuals against hour_diff
        plt.figure(figsize=(10, 6))
        plt.scatter(df["hour_diff"], df["residual"], color="blue", alpha=0.5)
        
        plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Reference line
        plt.xlabel("Hour Difference")
        plt.ylabel("Residual (clouds - a_clouds)")
        plt.title("Forecast Accuracy Over Time")
        plt.grid(True)
        plt.show()
        if cascade:
            plot_type =  'binned_scatter'
    
    if plot_type == 'binned_scatter' :
        # Bin data into groups
        df['hour_diff_bin'] = pd.cut(df['hour_diff'], bins=np.arange(0, df['hour_diff'].max(), 6))  # Adjust bin size
        
        # Compute mean and std for each bin
        binned_stats = df.groupby('hour_diff_bin')['residual'].agg(['mean', 'std'])
        
        # Plot with error bars
        plt.figure(figsize=(8,5))
        plt.errorbar(binned_stats.index.categories.mid, binned_stats['mean'], yerr=binned_stats['std'], fmt='o-', capsize=5)
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual (Predicted - Actual Cloud Cover)')
        plt.title('Mean Residuals & Variability Over Forecast Lead Time')
        plt.show()
        if cascade:
            plot_type =  'density_heat'
        
    elif plot_type == 'density_heat':
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Figures of merit
        if density_merits:
            df['absolute_error'] = abs(df['residual'])
            
            # Calculate Mean Absolute Error (MAE)
            MAE = df['absolute_error'].mean()
            print(f"Mean Absolute Error  for Temeprature (MAE): {MAE}")
            
            # Calculate Root Mean Squared Error (RMSE)
            RMSE = ( df["residual"] ** 2).mean() ** 0.5
            print(f"Root Mean Squared Error (RMSE): {RMSE}")
            
            ##MAPE
            ## Calculate Mean Absolute Percentage Error (MAPE)
            #MAPE = 100 * (df['absolute_error'] / df['a_cloud_cover']).mean()
            #print(f"Mean Absolute Percentage Error (MAPE): {MAPE}%")
            
            ##SMAPE
            #numerator = df['absolute_error']
            #denominator = (np.abs(df["clouds"] + df["a_cloud_cover"]))/ 2
            #with np.errstate(divide='ignore', invalid='ignore'):
                #smape_values = np.where(denominator == 0, 0, numerator / denominator)
            #print(f"SMAPE: {100 * np.mean(smape_values):.2f}%")
           
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      
        g = sns.JointGrid(data=df, x="hour_diff", y="residual", marginal_ticks=True)
        
        # Density Plot
        g.plot_joint(sns.kdeplot, fill=True,  cmap='PuOr')
        
        # Marginal Distributions (Histograms or KDEs)
        g.plot_marginals(sns.histplot, kde=True, bins=20)                                     
        
        ## X-axis bar plot showing standard deviation at each hour_diff interval
        # Compute standard deviation of residuals at each hour_diff interval
        #std_df = df.groupby("hour_diff")["residual"].std().reset_index()        
        #sns.barplot(data=std_df, x="hour_diff", y="residual", ax=g.ax_marg_x, color="darkblue", alpha=0.7)        
        ## Adjust Marginal X-Axis
        #g.ax_marg_x.set_ylabel("Std Dev of Residuals")  # Label for standard deviation
        #g.ax_marg_x.set_xlabel("")  # Remove duplicate X-label
        #g.ax_marg_x.grid(True, linestyle="--", alpha=0.5)  # Optional grid
        
        # Remove X-axis marginal to keep only Y
        g.ax_marg_x.set_visible(False)

        # Set title, X-label, and Y-label using Matplotlib
        g.figure.suptitle("Density Plot of Forecast Lead Time vs. Cloud Cover Residuals", fontsize=14)
        g.ax_joint.set_xlabel("Forecast Lead Time (hours)", fontsize=12)
        g.ax_joint.set_ylabel("Residuals (Forecast Clouds - Actual Clouds)", fontsize=12)
        plt.show()
        if cascade:
            plot_type =  'rolling_std'
        
    elif plot_type == 'rolling_std':
        df_sorted = df.sort_values('hour_diff')
        df['rolling_std'] = df_sorted['residual'].rolling(window=10, min_periods=5).std()
        
        plt.figure(figsize=(8,5))
        plt.plot(df_sorted['hour_diff'], df['rolling_std'], color='red', label='Rolling Std Dev')
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual Variability')
        plt.title('Expanding Error Over Forecast Lead Time')
        plt.legend()
        plt.show()       
        if cascade:
            plot_type =  'heat_vnm'
        
    elif plot_type == 'heat_vnm':
        # Bin the data
        df['hour_bin'] = pd.cut(df['hour_diff'], bins=np.arange(0, df['hour_diff'].max() + 6, 6))  # 6-hour bins
        df['residual_bin'] = pd.cut(df['residual'], bins=np.linspace(df['residual'].min(), df['residual'].max(), 20))  # Residual bins
        
        # Pivot table to compute mean residuals
        heatmap_data = df.pivot_table(index='residual_bin', columns='hour_bin', values='residual', aggfunc='mean')
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='PuOr', center=0, annot=False, cbar=True)
        
        plt.xlabel('Forecast Lead Time (Hours)')
        plt.ylabel('Residual Bins')
        plt.title('Heatmap of Mean Residuals Over Forecast Lead Time')
        plt.show()
        
    return
    
    
#-----------------------------------------------------------------------------------------------------
def distances_nodes_to_weather_station_dist(data_dir):
    '''
    Exaines the calculated distance values for the node lat and long
    vs. the responding weatherstation closets to those nodes. Provides
    a distribution of those distances. Could help provide error information
    concerning any examination of accuracy of forecasts vs. actual data.
    
    Parameters
    ---------------------------
    data_dir : str
         Path to location data files for historical weather are stored
        
    Results
    --------------------------
    A plot of the distribution.
    '''
    print ('Examining weather station vs node distances')
    # Get all CSV files in the folder
    csv_files = glob.glob(f"{data_dir}/*.csv")
    
    # Initialize a list to store the first distance values
    distance_values = []
    
    # Loop through each file and extract the first distance value
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Ensure the column 'distance' exists
        if 'distance_ws_to_node_km' in df.columns:
            # Append the first value of the 'distance' column
            distance_values.append(df['distance_ws_to_node_km'].iloc[0])

    # Plot the distribution using seaborn
    sns.histplot(distance_values, kde=True, bins=20, color='blue')
    plt.title('Distance of Node from Reporting Weather Station')
    plt.xlabel('Distance (km)')
    plt.ylabel('Frequency')
    plt.show()
    return

#-----------------------------------------------------------------------------------------------------
def plot_daily_forecast_for_node(df):
        
    # Convert dates to datetime
    df["predict_day"] = pd.to_datetime(df["predict_day"])
    df["forecast_day"] = pd.to_datetime(df["forecast_day"])
    
    # Initialize 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Encode dates as numbers for plotting
    df["forecast_day_num"] = mdates.date2num(df["forecast_day"])
    df["predict_day_num"] = mdates.date2num(df["predict_day"])
    
    # Plot min and max temperatures for each prediction day
    for predict_day in df["predict_day"].unique():
        subset = df[df["predict_day"] == predict_day]
        ax.plot(
            subset["forecast_day_num"],  # X-axis
            subset["temp_min"],          # Y-axis
            subset["predict_day_num"],   # Z-axis
            label=f"Min Temp (Predicted {predict_day.date()})",
            marker="o",
        )
        ax.plot(
            subset["forecast_day_num"],  # X-axis
            subset["temp_max"],          # Y-axis
            subset["predict_day_num"],   # Z-axis
            label=f"Max Temp (Predicted {predict_day.date()})",
            marker="o",
        )
    
    # Customize the axes
    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("Temperature (°C)")
    ax.set_zlabel("Prediction Day")
    
    # Format X-axis and Z-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.zaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    plt.xticks(rotation=45)
    
    # Rotate the view to set the desired orientation
    ax.view_init(elev=118, azim=-71, roll=20)  # Adjust elevation and azimuth for layout
    
    # Add a legend and title
    #ax.legend(loc="best")
    ax.set_title("3D Plot of Temperature Predictions")
    
    # Show the plot
    plt.show()
    return

#-----------------------------------------------------------------------------------------------------
def process_open_weather_json(json_data):
    '''
    Function to parse the JSON and extract relevant data
    '''
    # Extract the prediction day (first date in the file)
    prediction_day = datetime.strptime(json_data['list'][0]['dt_txt'], '%Y-%m-%d %H:%M:%S').date()
    # Initialize lists to store parsed data
    daily_data = []

    # Loop through the list of weather entries in the JSON
    for entry in json_data['list']:
        # Extract the datetime and temperature values
        dt_txt = entry['dt_txt']
        temp_min = entry['main']['temp_min']
        temp_max = entry['main']['temp_max']

        # Convert 'dt_txt' to a date
        forecast_day  = datetime.strptime(dt_txt, '%Y-%m-%d %H:%M:%S').date()

        # Append the data as a tuple
        daily_data.append((prediction_day, forecast_day, temp_min, temp_max))
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_data, columns=['predict_day', 'forecast_day', 'f_temp_min', 'f_temp_max'])

    # Group by 'Forecast_Day' to compute min/max for each day
    df_grouped = df.groupby(['predict_day', 'forecast_day']).agg({
        'f_temp_min': 'min',
        'f_temp_max': 'max'
    }).reset_index()
    df_grouped['f_temp_min'] = (df_grouped['f_temp_min'] - 273.15).round(2)
    df_grouped['f_temp_max'] = (df_grouped['f_temp_max'] - 273.15).round(2)

    return df_grouped

#-----------------------------------------------------------------------------------------------------
def process_open_weather_csv(df_forecast, start_date, end_date, forecast_period, max_forecast):
    '''
    Function to parse the CSV, extract  and process into relevant data for data 
    '''
    print (datetime.now().isoformat(sep=' ') + '     cleaning and preparing ow forecast dataframe ' )
    
    result = pd.DataFrame()

    #prepare dataframe for work
    df_forecast.rename(columns={
        'forecast dt iso': 'predict_time',
        'slice dt iso': 'forecast_time',
    }, inplace=True)
    #clean up UTC marker in forecast_time
    df_forecast['forecast_time'] = df_forecast['forecast_time'].str.replace(' UTC', '', regex=False)            
    # Convert `forecast_time` into a datetime object
    df_forecast['forecast_time'] = pd.to_datetime(df_forecast['forecast_time'])
    ##localize timestamps
    #df_forecast = sp.localize_forecast_dataframe(df_forecast)
    # Create a new column 'forecast_day' by extracting the date part from 'forecast_time'
    df_forecast['forecast_day'] = df_forecast['forecast_time'].dt.date    
    df_forecast["predict_day"] = df_forecast["predict_time"].dt.date
    #now remove any forecasts that are beyond the limit from the predict day
    # Filter out rows where the difference is more than 7 days
    df_forecast_max = df_forecast[(df_forecast['forecast_time'] - df_forecast['predict_time']).dt.days < max_forecast]
    
    #Drop hours field as not needed.
    df_forecast_max.drop('hours', axis=1, inplace=True)   
    
    #Cleanup columns to needed units
    #Convert temperature from Kelvin to Celcius
    df_forecast_max['temperature_C'] = (df_forecast_max['temperature'] - 273.15).round(2)
    #Delete (drop) the old temperature column
    df_forecast_max.drop(columns=["temperature"], inplace=True)
    #Rename the new column back to "temperature"
    df_forecast_max.rename(columns={"temperature_C": "temperature"}, inplace=True)    
    
    #Convert dew point from Kelvin to Celcius
    df_forecast_max['dew_point_C'] = (df_forecast_max['dew_point'] - 273.15).round(2)
    #Delete (drop) the old dew_point column
    df_forecast_max.drop(columns=["dew_point"], inplace=True)
    #Rename the new column back to "dew_point"
    df_forecast_max.rename(columns={"dew_point_C": "dew_point"}, inplace=True)       
    
    #reorder data_frame and remove unneeded columns
    #df_forecast_max = df_forecast_max[sp.dataframe_col_list]
     
    #Convert start and end date to datetime objects
    tz_string = sp.get_local_timezone_from_forecast_df(df_forecast)
    local_tz = timezone(tz_string)
    local_start_date = pd.Timestamp(start_date).replace(hour=0, minute=0, second=0, tzinfo=local_tz)
    local_end_date = pd.Timestamp(end_date).replace(hour=23, minute=59, second=59, tzinfo=local_tz)
    # Convert local times to UTC
    utc_start_date = local_start_date.astimezone(timezone("UTC"))
    utc_end_date = local_end_date.astimezone(timezone("UTC"))
    # Filter forecast_df based on UTC timestamps
    targeted_df = df_forecast_max[
        (df_forecast_max["predict_time"] >= utc_start_date) &
        (df_forecast_max["predict_time"] <= utc_end_date)
    ]
   
    ##Set start and end date to datetime objects
    #start_date = pd.to_datetime(start_date).tz_localize(df_forecast_max["forecast_time"].dt.tz)
    #end_date = pd.to_datetime(end_date + " 23:59:59").tz_localize(df_forecast_max["forecast_time"].dt.tz)

    ## Filter DataFrame based on predict_day+
    #targeted_df = df_forecast_max[(df_forecast_max['predict_time'] >= start_date) & (df_forecast_max['predict_time'] <= end_date)]
    
    #Begin rollingup forecast data by blocks basedon forecast_period
    if forecast_period > 24 or forecast_period < 1:
        raise ValueError("forecast_period must be between 1 and 24 hours.")
        
    # if sample is one hour, it is native for the file and it is just returned as is. 
    if forecast_period == 1:
        #Set to three decial places
        targeted_df = targeted_df.round(3)
        
        return targeted_df  
    
    # File is resampled to requested hourly interval. Best to choose even divisions of 24 hour period (6, 12, 24)
    else:
        resampled_list = []
        resample_period = str(forecast_period) + 'h'
        agg_dict = {
                    'predict_day' : ['first'], 
                    'forecast_day' : ['first'], 
                    'temperature': ['mean', 'max', 'min'],
                    'dew_point': ['mean', 'max', 'min'],
                    'humidity': ['mean', 'max', 'min'],
                    'pressure': ['mean', 'max', 'min'],
                    'ground_pressure': ['mean', 'max', 'min'],
                    'clouds' : ['mean', 'max', 'min'],
                    'wind_speed' : ['mean', 'max', 'min'],
                    'wind_deg' : [sp.circular_mean],
                    'rain' : ['mean', 'max', 'min'],
                    'snow' : ['mean', 'max', 'min'],
                    'ice' : ['mean', 'max', 'min'],
                    'fr_rain' : ['mean', 'max', 'min'],
                    'convective' : ['mean', 'max', 'min'],
                    'snow_depth' : ['mean', 'max', 'min'],
                    'accumulated' : ['mean', 'max', 'min'],
                    'rate' : ['mean', 'max', 'min'],
                    'probability' : ['mean', 'max', 'min']
                }
                
        for p_time, group_df in targeted_df.groupby('predict_time'):
            #reset index to fcailitate resample
            group_df = group_df.set_index('forecast_time')
            #Perfomr resampling based on resample_period and aggrgation dict 
            group_resampled = group_df.resample(resample_period).agg(agg_dict)
            #Keep track of which predict_time this came from
            group_resampled["predict_time"] = p_time
            # Append to a list of DataFrames
            resampled_list.append(group_resampled)
        
        #Bring all the predict_time groups back    
        agg_df =  pd.concat(resampled_list).reset_index()
        
        #Clean up column names
        agg_df.columns = sp.flatten_column_names(agg_df.columns)

        #rename wind_deg to wind_deg_c_mean
        agg_df = agg_df.rename(columns={'wind_deg': 'wind_deg_c_mean'})
       
        #Set to three decial places
        agg_df = agg_df.round(3)
        
        #Perform the aggregations
        return agg_df  
        
        


#-----------------------------------------------------------------------------------------------------
def extract_actual_weather_data(csv_dir, node_id, start_date, end_date):
    '''
    Open up a weather file from the weather site (NOAA)  on local drive.
    Create a dataframe from the node data  and filter it for the dates in question. Return
    the filtered data frame.
    
    Parameters
    ---------------------------------
    node_id : str
         WECC Node id defined as partof REGROW Project.
    start_date : str
         ISO 8601 beginning timestamp of temporal period to filter on
    end_date : str
         ISO 8601ending timestamp of temporal period to filter on
         
    Returns
    --------------------------------
    filtered_combined_weather_df : dataframe object
         iorganized and cleaned data frame of actual weather values for a node
         encompassing the temporal period in question.
    '''
    # Initialize a list to store the data
    weather_data = []

    # Loop through files in the directory
    for file_name in os.listdir(csv_dir):
        # Only process files that match the node_id prefix
        if file_name.startswith(node_id) and file_name.endswith(".csv"):
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Extract relevant columns: date, max_daily_temp, min_daily_temp, mean_daily_temp
            # Adjust column names if necessary based on your actual CSV format
            weather_data.append(df[['timestamp', 'a_temperature', 'a_cloud_cover', 'a_wind_speed','a_wind_dir']])
    
    # Combine all the data into a single DataFrame
    combined_weather_data = pd.concat(weather_data, ignore_index=True)
    
    # Convert the 'date' column to datetime to make merging easier
    combined_weather_data['timestamp'] = pd.to_datetime(combined_weather_data['timestamp'])
   
    # Filter DataFrame based on start and end_date. Add five to end date to get all in forecast periods
    end_date_dt = pd.to_datetime(end_date)
    buffered_date = end_date_dt + timedelta(days=5)
    filtered_combined_weather_df = combined_weather_data[(combined_weather_data['timestamp'] >= start_date) & (combined_weather_data['timestamp'] <= buffered_date)]
    
    return filtered_combined_weather_df

#-----------------------------------------------------------------------------------------------------
def extract_actual_weather_data_s3(s3, bucket, node_id, start_date, end_date):
    '''
    Open up a weather file from the weather site (NOAA)  download archive onS3.
    Create a dataframe from the node data  and filter it for the dates in question. Return
    the filtered data frame.
    
    Parameters
    ---------------------------------
    s3 : s3 client object
    bucket : str
         The name of the bucket where the data is stored
    node_id : str
         WECC Node id defined as partof REGROW Project.
    start_date : str
         ISO 8601 beginning timestamp of temporal period to filter on
    end_date : str
         ISO 8601ending timestamp of temporal period to filter on
         
    Returns
    --------------------------------
    filtered_combined_weather_df : dataframe object
         iorganized and cleaned data frame of actual weather values for a node
         encompassing the temporal period in question.
    '''
    # Initialize a list to store the data
    weather_data = []
    #  paths
    prefix_actual = "REGROW/actual_weather_data_for_nodes/2018-2022/"
    print (datetime.now().isoformat(sep=' ') + '     Locating historic actual station reported weather CSV files from S3 archive')
    csv_files = []
    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket, 'Prefix': prefix_actual}
    
    # Use paginator to retrieve all files
    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                
                # Filter for Open Weather historical forecast CSV files
                if file_key.endswith(".csv"):
                    csv_files.append(file_key)
    
    if not csv_files:
        print ('No CSV files found in working dir. Skipping node...')
        return
    else:
        tmp_f = csv_files[0].split('/')[-1]
        node_id = tmp_f.split('_')[0]            
        # Process each JSON file
        for csv_file in csv_files:
            df = sp.read_csv_from_s3(s3, bucket, csv_file)

            # Extract relevant columns: date, max_daily_temp, min_daily_temp, mean_daily_temp
            # Adjust column names if necessary based on your actual CSV format
            weather_data.append(df[['timestamp', 'a_temperature', 'a_cloud_cover', 'a_wind_speed','a_wind_dir']])
    
    # Combine all the data into a single DataFrame
    combined_weather_data = pd.concat(weather_data, ignore_index=True)
    
    # Convert the 'date' column to datetime to make merging easier
    combined_weather_data['timestamp'] = pd.to_datetime(combined_weather_data['timestamp'])
    
    #Make it a timetamp column
    combined_weather_data['timestamp'] = combined_weather_data['timestamp'].dt.tz_localize('UTC')
   
    # Filter DataFrame based on start and end_date. Add five to end date to get all in forecast periods
    end_date_dt = pd.to_datetime(end_date)
    buffered_date = end_date_dt + timedelta(days=5)
    filtered_combined_weather_df = combined_weather_data[(combined_weather_data['timestamp'] >= start_date) & (combined_weather_data['timestamp'] <= buffered_date)]
    
    return filtered_combined_weather_df
    
    
#-----------------------------------------------------------------------------------------------------
def process_ow_daily_forecast_files_for_min_and_max_temp(target_dir, use_s3=False):
    '''
    Opens up files in daily forecast S3 bucket and begins to work trhough each node forecast. It finds
    all values daily predict file and computes and min and a max value form each forecast day.
    
    Parameters
    -----------------------------
    target_dir : str
         path to the location for retrieving and storing results on local system if use_s3 is false.
         Only use path for storing reults files if use_s3 is true.
    use_s3: bool
         Use archives on S3 as point to pull files from
         
    Returns
    -------------------
    merged_df : dataframe object
         A dataframe containg all the daily min, mean, and max temps corresponding to each predict day
         and then each forecast in the predict day. Also containg the actual values for every day as
         read ffrom the actual weather data files.
    '''    
    all_data = []
    node_id = ''
    merged_df = pd.DataFrame()
    # Initialize the S3 client
    s3 = boto3.client('s3')
    # Bucket and paths
    bucket = "pvdrdb-transfer"
    prefix = "REGROW/working/"
    
    if use_s3:
        print ('Processing json files from S3 archive')
        json_files = []
        paginator = s3.get_paginator('list_objects_v2')
        operation_parameters = {'Bucket': bucket, 'Prefix': prefix}
        
        # Use paginator to retrieve all files
        for page in paginator.paginate(**operation_parameters):
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    
                    # Filter for JSON files
                    if file_key.endswith(".json"):
                        json_files.append(file_key)
        
        if not json_files:
            print ('No json files found in working dir. Skipping node...')
            return
        else:
            tmp_f = json_files[0].split('/')[-1]
            node_id = tmp_f.split('_')[0]            
            print ('Processing json files from S3 archive for node ' + node_id)
            # Process each JSON file
            for json_file in json_files:
                print ('Processing json file ' + json_file)
                json_data = sp.read_json_from_s3(s3, bucket, json_file)
                processed_df = process_open_weather_json(json_data)
                all_data.append(processed_df)
            
            # Combine all processed data into a single DataFrame
            final_df = pd.concat(all_data, ignore_index=True)
            
            #Find out difference between predict and forecast
            # Convert date columns to datetime objects
            final_df["predict_day"] = pd.to_datetime(final_df["predict_day"])
            final_df["forecast_day"] = pd.to_datetime(final_df["forecast_day"])
            
            # Calculate the mean temperature
            final_df["f_temp_mean"] = (final_df["f_temp_max"] + final_df["f_temp_min"]) / 2
            # Calculate the difference in days
            final_df["day_diff"] = (final_df["forecast_day"] - final_df["predict_day"]).dt.days            
            
            #Get Actual Weather data for node over period
            actual_weather_data_archive = 'C:/ReGROW/Historic_weather_data/forecast_compare'
            weather_data_df = extract_actual_weather_data(actual_weather_data_archive, node_id)
            
            # Convert 'forecast_day' to datetime for consistency
            final_df['forecast_day'] = pd.to_datetime(final_df['forecast_day'])
            merged_df = pd.merge(final_df, weather_data_df, left_on='forecast_day', right_on='date', how='left')
            
            # Drop the extra 'date' column that came from the merge
            merged_df.drop('date', axis=1, inplace=True)
            
            merged_df.rename(columns={
                'max_daily_temp': 'a_temp_max',
                'min_daily_temp': 'a_temp_min',
                'mean_daily_temp': 'a_temp_mean',
            }, inplace=True)
            
            # Add difference columns for each temperature type (forecast vs actual)
            merged_df['diff_temp_max'] = merged_df['f_temp_max'] - merged_df['a_temp_max']
            merged_df['diff_temp_min'] = merged_df['f_temp_min'] - merged_df['a_temp_min']
            merged_df['diff_temp_mean'] = merged_df['f_temp_mean'] - merged_df['a_temp_mean']           
            # Round the difference columns to four significant digits
            merged_df['diff_temp_max'] = merged_df['diff_temp_max'].apply(lambda x: round(x, 4))
            merged_df['diff_temp_min'] = merged_df['diff_temp_min'].apply(lambda x: round(x, 4))
            merged_df['diff_temp_mean'] = merged_df['diff_temp_mean'].apply(lambda x: round(x, 4))
            
             #Write out final CSV for node
            merged_df.to_csv(target_dir + '/' + node_id + '_forecast_vs_actual_processed.csv', index = False)
            #VIZ
            #plot_dist_forecast_to_actual_diffs(target_dir, node_id, merged_df)
 
            #Now create new column for node id so merged_df can be added to a master dataframe
            merged_df['node_id'] = node_id
     
     #local dirs
    else:
        # Get all JSON files in the folder
        json_files = glob.glob(f"{target_dir}/*.json")
        json_files = [file.replace("\\", "/") for file in json_files]
        
        # Process each JSON file
        for json_file in json_files:
            node_id = os.path.basename(json_file).split('_')[0]
            with open(json_file, 'r') as file:
                json_data = json.load(file)
                processed_df = process_open_weather_json(json_data)
                all_data.append(processed_df)
        
        # Combine all processed data into a single DataFrame
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(target_dir + '/' + node_id + '_forecast_to_actual.csv', index = False)
    
    #return completed data frame
    return merged_df


#-----------------------------------------------------------------------------------------------------
def process_ow_historical_forecast_files(target_dir, start_date, end_date, timezone_offset, forecast_period=24, max_forecast=7, merge_actual_weather=False):
    '''
    Opens up files of historical forecast S3 bucket and begins to work through each node forecast. It finds
    all values daily predict file and computes and min and a max value from each forecast day.
    Base granularity for the forecasts is one hour, but can be rolled up to 24 (daily) if needed
    
    Parameters
    -----------------------------
    target_dir : str
         path to the location for storing results on local system.
    start_date : str
         ISO 8601 date to define start period of prediction days in forecast data
    end_date : str
         ISO 8601 date to define the last day of the period of prediction days in forecast data
    timezone_offset : int
         The hour offset for the node and its timezone.
    forecast_period: int
         Over what hourly granularity is the forecast data to be aggrregated to. Maximum and default = 24.
    max_forecast : int
         Maximum number of days from predict day to use as forecasts. Default = 7
    merge_actual_weather : bool
         Indicate whether to merge in actual weather data or to just look at the forecast,. Default = False
         
    Returns
    -------------------
    merged_df : dataframe object
         A dataframe containg all the daily min, mean, and max temps corresponding to each predict day
         and then each forecast in the predict day. Also containg the actual values for every day as
         read ffrom the actual weather data files.
    '''    
    all_data = []
    node_id = ''
    merged_df = pd.DataFrame()
    # Initialize the S3 client
    s3 = boto3.client('s3')
    # Bucket and paths
    bucket = "pvdrdb-transfer"
    prefix_forecast = "REGROW/weather_forecast_data/working/"
    
    print (datetime.now().isoformat(sep=' ') + '     Locating historic Open Weather CSV files from S3 archive')
    csv_files = []
    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket, 'Prefix': prefix_forecast}
    
    # Use paginator to retrieve all files
    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                
                # Filter for Open Weather historical forecast CSV files
                if file_key.endswith(".csv"):
                    csv_files.append(file_key)
    
    if not csv_files:
        print ('No CSV files found in working dir. Skipping node...')
        return
    else:
        tmp_f = csv_files[0].split('/')[-1]
        node_id = tmp_f.split('_')[0]            
        # Process each JSON file
        for csv_file in csv_files:
            #Check if file has been processed and skip if so.
            if csv_file in sp.exclude_files:
                print (datetime.now().isoformat(sep=' ') + '    File ' + csv_file + ' already processed. Skipping...')
                continue            
            
            print (datetime.now().isoformat(sep=' ') + '     Loading file ' + csv_file + ' from S3 into dataframe ')
            hist_df = sp.read_csv_from_s3(s3, bucket, csv_file)
            print (datetime.now().isoformat(sep=' ') + '     File ' + csv_file + ' acquired into dataframe. Set up for processing.')
            # Remove the "UTC" text
            hist_df['forecast dt iso'] = hist_df['forecast dt iso'].str.replace(' UTC', '', regex=False)            
            #convert predict_day to timestamp
            hist_df['forecast dt iso'] = pd.to_datetime(hist_df['forecast dt iso'])
            #if a start or end date are not provided then use min and max from file.
            if not start_date:
                start_date =  hist_df['forecast dt iso'].min()
            if not end_date:
                end_date =  hist_df['forecast dt iso'].max()
            #Create date tags
            start_tag = start_date.date().strftime('%Y%m%d')
            end_tag = end_date.date().strftime('%Y%m%d')
            
            #process file for needed forecast data
            processed_df = process_open_weather_csv(hist_df, start_date, end_date, forecast_period, max_forecast)       
        
            # Convert date columns to datetime objects
            processed_df["predict_day"] = pd.to_datetime(processed_df["predict_day"])
            processed_df["predict_day"] = processed_df["predict_day"].dt.date
            processed_df["forecast_day"] = pd.to_datetime(processed_df["forecast_day"])
            processed_df["forecast_day"] = processed_df["forecast_day"].dt.date
            
            #remove any rows where forecast_time is less than predict_time
            final_df = processed_df[processed_df['forecast_time'] >= processed_df['predict_time']].copy()
            
            # Calculate the difference in days between foreast and predict
            final_df["day_diff"] = (final_df["forecast_day"] - final_df["predict_day"]).apply(lambda x: x.days)
            #calculate differenc in hours between forrecast and predict
            final_df['hour_diff'] = ((final_df['forecast_time'] - final_df['predict_time']) / np.timedelta64(1, 'h')).round().astype(int)
             
            #Sort dataframe
            final_df.sort_values(by=["predict_day", "forecast_time", "day_diff", "hour_diff"],
                           ascending=[True, True, True, True],
                           inplace=True)
            timezone_string = final_df['forecast_time'].iloc[0].tz
            #Test DEBUG
            #final_df.to_csv('C:/ReGROW/final_df.csv')
            #END DEBUG
            
            if merge_actual_weather:                    
                if forecast_period != 1:
                    # Drop the extra 'date' column that came from the merge
                    merged_df.drop('date', axis=1, inplace=True)
                    
                    merged_df.rename(columns={
                        'max_daily_temp': 'a_temp_max',
                        'min_daily_temp': 'a_temp_min',
                        'mean_daily_temp': 'a_temp_mean',
                    }, inplace=True)
                
                    # Add difference columns for each temperature type (forecast vs actual)
                    merged_df['diff_temp_max'] = merged_df['f_temp_max'] - merged_df['a_temp_max']
                    merged_df['diff_temp_min'] = merged_df['f_temp_min'] - merged_df['a_temp_min']
                    merged_df['diff_temp_mean'] = merged_df['f_temp_mean'] - merged_df['a_temp_mean']           
                    # Round the difference columns to four significant digits
                    merged_df['diff_temp_max'] = merged_df['diff_temp_max'].apply(lambda x: round(x, 4))
                    merged_df['diff_temp_min'] = merged_df['diff_temp_min'].apply(lambda x: round(x, 4))
                    merged_df['diff_temp_mean'] = merged_df['diff_temp_mean'].apply(lambda x: round(x, 4))
                
                # Hourly data
                else: 
                    #Get Actual Weather data for node over period
                    weather_data_df = extract_actual_weather_data_s3(s3, bucket, node_id, start_date, end_date)                                        
                    merged_df = pd.merge(final_df, weather_data_df, left_on='forecast_time', right_on='timestamp', how='left')
                
                #localize this
                merged_df = sp.localize_forecast_dataframe(merged_df)
                #Now create new column for node id so merged_df can be added to a master dataframe
                merged_df['node_id'] = node_id
  
                #Filter to final columns and positions
                composite_df = merged_df[sp.merged_dataframe_col_list]
                
                #Write out final CSV for node
                composite_df.to_csv(target_dir + '/' + node_id + '_forecast_vs_actual_processed_' + start_tag + '_' + end_tag + '.csv', index = False)
               
                #return completed data frame
                return composite_df           

            else:   # Not merging forecast with actual weather
                #Now create new column for node id
                final_df['node_id'] = node_id
                #set final column order
                if forecast_period == 1:
                    final_df = final_df[sp.dataframe_col_list]
                else:
                    final_df = final_df[sp.agg_dataframe_col_list]
                
                if forecast_period == 24:
                    period_tag = 'daily_'
                else:
                    period_tag = str(forecast_period) + 'h_'
                
                #Write out final CSV for node
                final_df.to_csv(target_dir + '/' + node_id + '_forecast_processed_' + period_tag +  start_tag + '_' + end_tag + '.csv')
                return final_df         
            
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
#                                                        MAIN                                                               #
# -------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------#
def main():
    # Get passed in args
    parser = argparse.ArgumentParser()
    #General parameters
    parser.add_argument('-n', nargs='?', type=str, help="Node ID")
    parser.add_argument('-pd', nargs='?', type=str, help="Prediction date. Used for comparing residuals based on when prediction made.")
    parser.add_argument('-all', help="Perform action on all nodes", action="store_true")
    #File information for accessing or storing
    parser.add_argument('-f', nargs='?', type=str, help="Harvest target file.")
    parser.add_argument('-d', nargs='?', type=str, help="Directory to write any plot files to")
    parser.add_argument('-dws', nargs='?', type=str, help="Data directory for weather station files")
    parser.add_argument('-dowf', nargs='?', type=str, help="Data directory for targeted Open Weather Forecast files")    
    parser.add_argument('-dawf', nargs='?', type=str, help="Data directory for targeted Actual Weather data files")    
    parser.add_argument('-s3', help="Use S3 archives to access data", action="store_true")
    #Data types
    parser.add_argument('-dt', nargs='?', type=str, help="Weather data type to operate with (temperature, wind_speed, or clouds) ")   
    parser.add_argument('-temp', help="Plot min and max forecast temp from file", action="store_true")
    parser.add_argument('-wind', help="Plot wind and gusts forecast", action="store_true")
    parser.add_argument('-clouds', help="Plot cloud cover forecast", action="store_true")
    #Access and work with daily harvests
    parser.add_argument('-f2a', help="Compare node forecast to actual", action="store_true") #OBSOLETE
    parser.add_argument('-nws', help="Use NWS for temp forecast", action="store_true") #OBSOLETE
    #Historical processing
    parser.add_argument('-hf2a', help="Compare historical node forecast to actual", action="store_true")
    parser.add_argument('-start', nargs='?', type=str, help="Start date for date range in examining historical files")    
    parser.add_argument('-end', nargs='?', type=str, help="End date for date range in examining historical files")    
    parser.add_argument('-period', nargs='?', type=int, help="Hourly forecast period aggrgation (Default 1 and Max 24)")    
    parser.add_argument('-max', nargs='?', type=int, help="Maximum days to forecast (Default=7, Max=10)")    
    parser.add_argument('-mnsrdb', help="Merge node nsrdb", action="store_true")
    #Visualizations
    parser.add_argument('-f2av', help="visualize the daily harvest actual to forecast predictions", action="store_true")  #OBSOLETE
    parser.add_argument('-pcomp', help="Compare predict time forecasts to each other for a type", action="store_true")
    parser.add_argument('-ceval', help="Evaluate cloud forecast for errors", action="store_true")
    parser.add_argument('-cws', help="Evaluate wind speed forecast for errors", action="store_true")
    parser.add_argument('-fullcomp', help="Full comparisons based on composite node file.", action="store_true")
    parser.add_argument('-conus', help="Add Conus hub height data to visualization", action="store_true")
    parser.add_argument('-nsrdb', help="Add NSRDB data to visualization", action="store_true")
    parser.add_argument('-norm', help="Normalize ground to hub and visualize", action="store_true")
     #Support worker methods
    parser.add_argument('-an2ws', help="Plot distribution of actual weather stations to nodes distances", action="store_true")
    parser.add_argument('-composite', help="Create composite png plot from plot pngs", action="store_true")
    parser.add_argument('-anim', help="Animate plots", action="store_true")
    
    #examples:
    #animate frames:
    #-anim
    
    # Process data roll up
    #-hf2a -dowf 'C:/ReGROW/' -period 1 -max 5     #BASIC REGROW DEFAULT: 
                                                                                               # Everything from start to finish in raw file at 1 
                                                                                               # hour res and five day forecast window.
    #-hf2a -dowf 'C:/ReGROW/' -start '2020-08-01' -end '2020-08-31' -period 1 -max 5     #Targeted across specific time
   
    # compare predict day forecasts
    #-pcomp -n '9q9wtp' -pd '2020-08-02' -dt temperature -dowf 'C:/ReGROW/' -f '9q9wtp_forecast_vs_actual_processed_20200801_20200831.csv'
    #-pcomp -n '9q9wtp' -pd '2020-08-14' -dt wind_speed -dowf 'C:/ReGROW/' -f '9q9wtp_forecast_vs_actual_processed_20200801_20200831.csv'

    #Compare forecast, model (NSRDB), and actual (NOAA) and plot comparison time series with forecast scatter and median
    #-f 'C:/ReGROW/target_nodes_forecast_to_actual/9qcbq0_forecast_vs_actual_processed_20180101_20221231.csv' -mnsrdb -dt temperature -period 6
    #-f 'C:/ReGROW/target_nodes_forecast_to_actual/9qcbq0_forecast_vs_actual_processed_20180101_20221231.csv' -mnsrdb -dt clouds
    
    #Explore weather value
    # -cws -f "REGROW/Forecast_and_Actual_Weather_Merged/9q6tde_forecast_vs_actual_processed_20180101_20221231.csv" -start 2020-08-09 00:00:00 -end 2020-08-23 23:59:59
    
    #process forecast and actual files to merge in nsrdb and conus
    #-mnsrdb -conus -all -start '2018-01-01 00:00:00' -end '2022-12-31 23:59:59'
    
    #Compare hub height with modled, actual, and forecast
    #-s3 -f "REGROW/Forecast_and_Actual_Weather_Merged/With_NSRDB_and_Conus/9q9hq4_forecast_actual_nsrdb_conus_2018-01-01-2022-12-31.csv" -fullcomp -conus -period 1 -dt wind_speed
        
    args = parser.parse_args()
    
    #Do comparison of a predict day's forecasts on Open Weather (4 per day) against actual weather 
    # for same period. Two plots: Non-scatter does line plots actual vs. scatter for each predict time, and
    # heat map of residuals. Regres does a regression plot for a single predict time (currently hardcoded
    # in method) of a predict day 
    if args.pcomp:
        print ('Initiating comparison of processed file predict times' )
        master_df = pd.read_csv(args.dowf + args.f)        
        compare_predict_time_forecasts(args.n, master_df, args.pd, args.dt, regres=True, target_dir=args.d)
        print ()
        quit()
    
    # Method to stack, temp, wind_speed, and clouds plot as a single image.
    #Files to stack are currently hardcoded in method.
    elif args.composite:
        create_composite_plots()
        quit()

    #String together a series of png plot images into an animation. Works for
    #single plots or composite above. Files to animate are currently hardcoded 
    #in method.
    elif args.anim:
        animate_plots()
        quit()
    
   #Plot Forecast vs actual basic. 
    if args.f2av:
        print ('Using harvest  config file: ../config/nodes.csv' )
        master_df = pd.read_csv(args.dowf + '/ow_forecast_vs_actual_master.csv')
        plot_scatters(master_df, args.dowf)

    #Process Weather forecast and actual data to create a master file.
    # Once done file is then stored on S3, so no need to re-run unless change in
    #input resources
    if args.f2a or args.hf2a:
        print ('Using harvest  config file: ../config/nodes.csv' )
        df_config = sp.read_config( '../config/nodes.csv')
        node_list = df_config['geocode'].tolist()
        #Loop through each geocode and find associated files in S3 bucket to extract
        master_df = pd.DataFrame()
        for node in node_list:
            #---For daily harvested forecasts---
            if args.f2a:
                #Find raw forecast file and copy to working point
                sp.search_and_copy_node_forecast(node, False)
                #Now began to process all files for this node mform the daily subdirectories.
                node_df = process_ow_daily_forecast_files_for_min_and_max_temp(args.dowf, use_s3=args.s3)
                #create a master dataframe concatenated from all files
                master_df = pd.concat([master_df, node_df], ignore_index=True)
            #---For historical OW forecasts---. 
            # Process forecasts to limit number of forecast days from predict. 
            # Change granularity of default of 1 hour resolution to daily. 
            elif  args.hf2a:
                #Find raw forecast file and copy to working point
                sp.search_and_copy_node_forecast(node, True)
                start_date = args.start if args.start else None
                end_date = args.end if args.end else None
                max_forecast_days = args.max if args.max else 7                   # Number of days from predict day to include forecast
                granularity_forecast = args.period if args.period else 24       #granularity default is 24 hours per forecast
                timezone_offset = int(df_config.loc[df_config["geocode"] == node, "tzoffset"].values)
                #based onparameters above filter file down to needed form.
                node_df = process_ow_historical_forecast_files(args.dowf,
                                                     start_date,
                                                     end_date,
                                                     timezone_offset, 
                                                     granularity_forecast,
                                                     max_forecast_days,
                                                     merge_actual_weather=True)
            #Concatenate into a master file
            master_df = pd.concat([master_df, node_df], ignore_index=True)
       
        #Write out master_df once all nodes complete
        if args.s3:
            column_order = ['node_id',
                            'predict_day',
                            'forecast_day',
                            'day_diff',
                            'f_temp_max',
                            'f_temp_min',
                            'f_temp_mean', 
                            'a_temp_max',
                            'a_temp_min',
                            'a_temp_mean', 
                            'diff_temp_max', 
                            'diff_temp_min', 
                            'diff_temp_mean',
                            'distance_ws_to_node_km',
                            'weather_station_coords'
                            ]
            # Rearrange the DataFrame columns
            master_df = master_df[column_order]        
            master_df.to_csv(args.dowf + '/ow_forecast_vs_actual_master.csv', index=False)
            plot_dist_forecast_to_actual_diffs(args.dowf, '', master_df)
 
        print ('Node actual to forecast weather analysis complete.')    
    
    #Merging in NSRDB files to actual and forecast, Once done file is then stored on S3
    elif args.mnsrdb:
        s3 = boto3.client('s3')        
        # Load CSV into DataFrame
        print (f"Processing NSRDB (and Conus Files)")
        #Run the prep work for all nodes
        if args.all:
            print("Scanning all node files")
            df_config = sp.read_config( '../config/nodes.csv')
            node_list = df_config['geocode'].tolist()
            for node in node_list:
                base_forecast_actual_file =  'REGROW/Forecast_and_Actual_Weather_Merged/' + node + '_forecast_vs_actual_processed_20180101_20221231.csv'
                print (f"Reading in node file {base_forecast_actual_file}")
                df = sp.read_csv_from_s3(s3, 'pvdrdb-transfer', base_forecast_actual_file)
                if args.conus:
                    #Load NSRDB and Conus into data frame
                    nsrdb_conus_file = 'REGROW/nsrdb_conus_data/' + node + '.csv'
                    df_nsrdb =  sp.read_csv_from_s3(s3, 'pvdrdb-transfer', nsrdb_conus_file)
                else:
                    #Load NSRDB file into dataframe.
                    nsrdb_file = 'REGROW/nsrdb-wecc-nodes/' + node + '.csv'
                    df_nsrdb =  sp.read_csv_from_s3(s3, 'pvdrdb-transfer', nsrdb_file)
                #Merge data into single file
                df_merged = merge_nsrdb(node, df, df_nsrdb, args.conus)
                start_tag = args.start.split(" ")[0]
                end_tag = args.end.split(" ")[0]
                file_key = 'REGROW/Forecast_and_Actual_Weather_Merged/With_NSRDB_and_Conus/' + node + '_forecast_actual_nsrdb_conus_' + start_tag + '-' + end_tag + '.csv'
                sp.write_df_to_s3(s3, "pvdrdb-transfer", file_key, df_merged)
        
        #One shot
        else:   
            print (f"Reading in node file { args.f}")
            df = pd.read_csv(args.f)
            node_id =df["node_id"].iloc[0]
            s3 = boto3.client('s3')
            if args.conus:
                #Load NSRDB and Conus into data frame
                nsrdb_conus_file = 'REGROW/nsrdb_conus_data/' + node_id + '.csv'
                df_nsrdb =  sp.read_csv_from_s3(s3, 'pvdrdb-transfer', nsrdb_conus_file)
            else:
                #Load NSRDB file into dataframe.
                nsrdb_file = 'REGROW/nsrdb-wecc-nodes/' + node_id + '.csv'
                df_nsrdb =  sp.read_csv_from_s3(s3, 'pvdrdb-transfer', nsrdb_file)
            #Merge data into single file
            df_merged = merge_nsrdb(node_id, df, df_nsrdb, args.conus)
            start_tag = args.start.split(" ")[0]
            end_tag = args.end.split(" ")[0]
            file_key = 'REGROW/Forecast_and_Actual_Weather_Merged/With_NSRDB_and_Conus/' + node_id + '_forecast_actual_nsrdb_conus_' + start_tag + '-' + end_tag + '.csv'
            sp.write_df_to_s3(s3, "pvdrdb-transfer", file_key, df_merged)
           
        quit()
        
    #Do direct comparisonbetween for ecast, actual, NSRDB (and Conus)
    #with line plots.
    elif args.fullcomp:
        print (f"Reading in node file {args.f}")
        if args.s3:
            s3 = boto3.client('s3')
            df = sp.read_csv_from_s3(s3, "pvdrdb-transfer", args.f)
        else:
            df = pd.read_csv(args.f)
        node_id =df["node_id"].iloc[0]
        if args.conus:
            processed_df = compare_linear_ground_vs_hub(node_id, df, args.period, data_type=args.dt)
            norm_df = pd.DataFrame()
            ret_df = normalize_linear_ground_vs_hub(node_id,
                                           processed_df,
                                           norm_df, 
                                           args.dt,
                                           line_plot = True, 
                                           regress_plot = False
                                           )                
            normalize_linear_ground_vs_hub(node_id,
                                           df,
                                           ret_df, 
                                           args.dt,
                                           line_plot = False, 
                                           regress_plot = True
                                           )                
        else:
            compare_linear_forecast_model_actual(node_id,
                                                 df,
                                                 args.period,
                                                 data_type=args.dt,
                                                 show_forecast_scatter=False, 
                                                 show_forecast_median=False,
                                                 show_noaa=True,
                                                 show_nsrdb=args.nsrdb, 
                                                 show_conus = args.conus
                                                 )
        quit()
    
    #Do direct comparison between forecast, actual, NSRDB (and Conus)
    #with line plots.
    elif args.norm:
        norm_df = pd.DataFrame()
        print (f"Reading in node file {args.f}")
        if args.s3:
            s3 = boto3.client('s3')
            df = sp.read_csv_from_s3(s3, "pvdrdb-transfer", args.f)
        else:
            df = pd.read_csv(args.f)
        node_id =df["node_id"].iloc[0]
        normalize_linear_ground_vs_hub(node_id,
                                       df,
                                       norm_df, 
                                      args.dt,
                                       #target_dir='C:/ReGROW/target_nodes_forecast_to_actual/Compare_hub_height_to_ground/',
                                       regress_plot = True
                                       )
        quit()

    # Compute distance from  node center to the reporting weather station .   
    elif args.an2ws:
        distances_nodes_to_weather_station_dist(args.dws)
    
    #Explore accuracy of each weather type based on processed  file
    #TEMP
    elif args.temp:        
        #Using aggregate dataframe from csv
        print (f"Reading in node file { args.f}")
        df = pd.read_csv(args.f)
        explore_accuracy_forecast_temp_vs_actual(df, True)
    
    #WIND
    elif args.wind:
        # Load the Forecast JSON data
        with open(args.f, 'r') as file:
            data = json.load(file)
        explore_sample_forecast_winds(data)
        
    #CLOUDS
    elif args.clouds:
        # Load CSV into DataFrame
        print (f"Reading in node file { args.f}")
        df = pd.read_csv(args.f)
        explore_sample_forecast_clouds_vs_actual_combined_df (df, True)
    
    elif args.ceval:
        # Load CSV into DataFrame
        print (f"Reading in node file { args.f}")
        df = pd.read_csv(args.f)
        explore_accuracy_forecast_clouds_vs_actual(df, True)
     
    elif args.cws:
        # Load CSV into DataFrame
        print (f"Reading in node file { args.f}")
        s3 = boto3.client('s3')
        df = sp.read_csv_from_s3(s3, "pvdrdb-transfer", args.f)
        #df = pd.read_csv(args.f)
        #explore_accuracy_forecast_windspeed_vs_actual (df, True)
        explore_accuracy_forecast_clouds_vs_actual(df,
                                                       args.start,
                                                       args.end, 
                                                       plot_type='density_heat',
                                                       cascade=True,
                                                       density_merits=True, 
                                                       compute_residuals = True)
    quit()


####################################################
if __name__ == '__main__':
    print ('..: Starting REGROW Forecast Data Analyzer :..')
    main()

