# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#		    		support.py
#
# A series of generic methods useful across the data harvest application
#              -------------------
# author 	     : Robert White
# date         	 : Jun 17, 2024
# copyright   : (C) 2024 by  NREL
# email          : robert.white@nrel.gov
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#    
#    Contact Information: 
#    robert.white@nrel.gov
#    Robert White  - Research Operations
#    National Renewable Energy Laboratory
#    15253 Denver West Parkway  MS3219  Golden, CO 80401-3305 
#
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import boto3
import pprint
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
from math import radians, sin, cos, sqrt, atan2
from io import StringIO
import pytz
from timezonefinder import TimezoneFinder
import s3fs

#Config locations
archive_dir='C:/ReGROW/Forecast_Data/'                  # Location path software is located in
config_filename = ''         #Name of configuration file

# Regex for YYYY-MM-DD hh:mm:ss(.sss)?+HH:MM
iso_space_offset_pattern = re.compile(
    r'^(\d{4}-\d{2}-\d{2}) '            # YYYY-MM-DD and a space
    r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)'    # hh:mm:ss(.sss)?
    r'([+\-]\d{2}:\d{2})$'             # +HH:MM or -HH:MM
) 

# list of files to ignore so staggered processing can be done.
exclude_files = []

dataframe_col_list = [
    'node_id', 
    'predict_day',
    'predict_time',
    'forecast_day',
    'forecast_time',
    'day_diff', 
    'hour_diff', 
    'temperature',
    'dew_point',
    'pressure',
   'ground_pressure',
   'humidity',
   'clouds',
   'wind_speed',
   'wind_deg',
   'rain',
   'snow',
   'ice',
   'fr_rain',
   'convective',
   'snow_depth',
   'accumulated',
   'rate',
   'probability'
]

merged_dataframe_col_list = [
    'node_id', 
    'predict_day',
    'predict_time',
    'forecast_day',
    'forecast_time',
    'day_diff', 
    'hour_diff', 
    'temperature',
    'dew_point',
    'pressure',
   'ground_pressure',
   'humidity',
   'clouds',
   'wind_speed',
   'wind_deg',
   'rain',
   'snow',
   'ice',
   'fr_rain',
   'convective',
   'snow_depth',
   'accumulated',
   'rate',
   'probability', 
   'a_temperature',
   'a_cloud_cover',
   'a_wind_speed',
   'a_wind_dir'
]


agg_dataframe_col_list = [
    'node_id', 
    'predict_day',
    'predict_time',
    'forecast_day',
    'forecast_time',
    'day_diff',
    'hour_diff', 
    'temperature_mean',
    'temperature_min',
    'temperature_max',
    'dew_point_mean',
    'dew_point_min',
    'dew_point_max',
    'humidity_mean',
    'humidity_min',
    'humidity_max',
    'pressure_mean',
    'pressure_min',
    'pressure_max',
   'ground_pressure_mean',
   'ground_pressure_min',
   'ground_pressure_max',
   'clouds_mean',
   'clouds_min',
   'clouds_max',
   'wind_speed_mean',
   'wind_speed_min',
   'wind_speed_max',
   'wind_deg_c_mean',
   'rain_mean',
   'rain_min',
   'rain_max',
   'snow_mean',
   'snow_min',
   'snow_max',
   'ice_mean',
   'ice_min',
   'ice_max',
   'fr_rain_mean',
   'fr_rain_min',
   'fr_rain_max',
   'convective_mean',
   'convective_min',
   'convective_max',
   'snow_depth_mean',
   'snow_depth_min',
   'snow_depth_max',
   'accumulated_mean',
   'accumulated_min',
   'accumulated_max',
   'rate_mean',
   'rate_min',
   'rate_max',
   'probability_mean',
   'probability_min',
   'probability_max'
]

# ---------------------------------------------------------------------------------------
def p_print(message):
    '''
    Method to print a list or dictionary out in easy readable form used for debugging
    '''
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(message)
    return


# ---------------------------------------------------------------------------------------
def read_config (path_and_file):
    '''
    access and opens the harvest targets for weather data from the Open Weather
    site and it is based on the "nodes" identified fro Regrow project
  
    Parameters
    ----------------------
    path_and_file : string
         location and filename of the configuration file 
         
    Returns
    ---------------------
    df : dataframe
         Composite dataframe of the configuration Information
    '''
    df = pd.read_csv(path_and_file)
    return df


# ---------------------------------------------------------------------------------------
def make_daily_dir ():
    '''
    Makes a dialy data folder inthe archives to store new weather data
    
    Parameters
    ----------------------
         
    Returns
    ---------------------
    '''
    # Get today's date
    today = datetime.date.today()
    
    # Format the date as YYYYMMDD
    formatted_date = today.strftime('%Y%m%d')
    
    # Create a directory with the formatted date as the name
    directory_name = archive_dir + formatted_date
    if os.path.isdir(directory_name):
        return directory_name
    else:
        os.makedirs(directory_name)
        print(f"Created directory: {directory_name}")
        return directory_name
    

# ---------------------------------------------------------------------------------------
def coordinates_delta_km (lat1, lon1, lat2, lon2):
    '''
    Method to calculate distance in km between two
    target coordinate pairs.Calculation uses the
    Haversine formula
    
    Parameters
    ----------------------
         
    Returns
    ---------------------
    '''
    # Earth's radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c   
    return round(distance, 4)


#-----------------------------------------------------------------------------------------------------
def read_json_from_s3(s3, bucket, key):
    '''
    Parameters
    ----------------------
         
    Returns
    ---------------------    
    '''
    try:
        # Fetch the file from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        file_content = response['Body'].read().decode('utf-8')
        
        # Parse the JSON data
        json_data = json.loads(file_content)
        return json_data
    except Exception as e:
        print(f"Error reading JSON from S3: {e}")
        return None

#-----------------------------------------------------------------------------------------------------
def read_csv_from_s3(s3, bucket_name, file_key):
    '''    
    Reads a CSV file from an S3 bucket into a pandas DataFrame.
    
    Parameters
    ------------------------------
    s3 : s3 boto3 client
        Connection to s3 from calling function
   bucket_name : str
        Name of the S3 bucket.
    key : str
         Key (path) to the CSV file in the bucket.
        
    Returns
    ------------------
    df : pandas.DataFrame object
         DataFrame containing the CSV data.
    '''    

    # Fetch the object from S3
    print(f"Reaching into S3 bucket {bucket_name} to access file {file_key} ")
    response = s3.get_object(Bucket=bucket_name, Key=file_key)

    # Read the CSV content into a pandas DataFrame
    csv_content = response['Body'].read().decode('utf-8')
    try:
        df = pd.read_csv(StringIO(csv_content))
    except:
        print (f"Unable to access S3 bucket {bucket_name} file {file_key} ")
    else:
        print (f"File read and processed into dataframe")
    
    return df

#-----------------------------------------------------------------------------------------------------
def write_df_to_s3(s3, bucket_name, file_key, df):
    '''    
    Reads a CSV file from an S3 bucket into a pandas DataFrame.
    
    Parameters
    ------------------------------
    s3 : s3 boto3 client
        Connection to s3 from calling function*
   bucket_name : str
        Name of the S3 bucket.
    key : str
         Key (path) to the CSV file in the bucket.
    df : dataframe object
         Data frame object to write as a csv
        
    Returns
    ------------------
    df : pandas.DataFrame object
         DataFrame containing the CSV data.
    '''    

    # Fetch the object from S3
    print(f"Feeding into S3 bucket {bucket_name} the file {file_key} ")
    target = "s3://" + bucket_name + "/" + file_key
    
    # Write the CSV content into S3 bucket
    try:
        df.to_csv(target, index=False)
    except:
        print (f"Unable to access S3 bucket {bucket_name}  or write file {file_key} ")
    else:
        print (f"File was written to S3  bucket {bucket_name}")
    
    return df
#-----------------------------------------------------------------------------------------------------
def read_csv_with_substring_from_s3(s3, bucket_name, substring):
    '''
    Searches all objects in the given S3 bucket for keys containing `substring`.
    If found, reads the first match into a Pandas DataFrame and returns it.
    
    Parameters
    ------------------------------
    s3 : s3 boto3 client
        Connection to s3 from calling function
   bucket_name : str
        Name of the S3 bucket.
    substring : str
         Key (path) to the CSV file in the bucket contins first part of filename.
        
    Returns
    ------------------
    df : pandas.DataFrame object
         DataFrame containing the CSV data.
    '''    
    # List all objects in the bucket
    # Note: If your bucket has a large number of objects, you should handle pagination.
    response = s3.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' not in response:
        print("No objects in bucket or unable to list objects.")
        return None
    
    # Look for the file whose key contains the specified substring
    for item in response['Contents']:
        key = item['Key']
        if substring in key:
            print(f"Found a match: {key}")

            # Retrieve the CSV object
            csv_obj = s3.get_object(Bucket=bucket_name, Key=key)
            
            # Read the object's body (bytes), decode to string
            body = csv_obj['Body'].read().decode('utf-8')
            
            # Use StringIO so pandas can read the string as file-like object
            df = pd.read_csv(StringIO(body))
            return df
    
    # If we reach here, nothing was found
    print(f"No objects found containing substring '{substring}'.")
    return None

#-----------------------------------------------------------------------------------------------------
def search_and_copy_node_forecast (node_name, use_historical=False):
    '''
    Finds the target file for the node forecast and copies it into the working directory (on S3)
    to be used in other processes.
    
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
    # Get all JSON or CSV (historical) files in the folder
    # Initialize the S3 client
    s3 = boto3.client('s3')
    
    # Bucket and paths
    bucket_name = "pvdrdb-transfer"
    #Use archived historical forecast
    if use_historical:
        source_prefix = "REGROW/weather_forecast_data/historic_forecast_data/raw"
    else: # Use daily forecast harvests
        source_prefix = "REGROW/weather_forecast_data/daily_harvests"
    destination_prefix = "REGROW/weather_forecast_data/working/"
    
    # Match date subfolders and file pattern
    working_pattern = fr"{destination_prefix}{node_name}_\d+_\d+_\-?\d+_\d+_\d+_hfsb_v1\.csv"
    match_pattern = fr"{source_prefix}/{node_name}_\d+_\d+_\-?\d+_\d+_\d+_hfsb_v1\.csv"

    # List all files in the folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=destination_prefix)
    
    # Check if there are any files in the folder
    if 'Contents' in response:
        print('Cleaning files from ' + destination_prefix)
        # Loop through the files and delete them
        for obj in response['Contents']:
            file_key = obj['Key']
            # Delete the file
            if file_key != 'REGROW/working/':
                # Match date subfolders and file pattern
                if re.match(working_pattern, file_key):
                    print(f"Node file already in bucket: {file_key}")
                    return #No need to go further.
                else:
                    s3.delete_object(Bucket=bucket_name, Key=file_key)
                    #print(f"Deleted: {file_key}")
            else:
                print("No files found in the working folder.")   
    
    #begin copy of files
    try:
        paginator = s3.get_paginator('list_objects_v2')
        operation_parameters = {'Bucket': bucket_name, 'Prefix': source_prefix}
        
        # Use paginator to retrieve all files
        for page in paginator.paginate(**operation_parameters):
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                file_key = obj['Key']
                # Match date subfolders and file pattern
                if re.match(match_pattern, file_key):
                    print(f"Found matching file: {file_key}")
                    
                    # Extract the file name
                    file_name = file_key.split('/')[-1]

                    # Define the destination key
                    destination_key = f"{destination_prefix}{file_name}"

                    # Copy the file to the destination
                    s3.copy_object(
                        Bucket=bucket_name,
                        CopySource={'Bucket': bucket_name, 'Key': file_key},
                        Key=destination_key
                    )
                    print(f"Copied file to {destination_key}")

    except Exception as e:
        print(f"An error occurred: {e}")
    return


#----------------------------------------------------------------------------------------
def get_local_timezone_from_forecast_df (df):
    '''
    '''
    # Initialize TimezoneFinder to get timezone info
    tf = TimezoneFinder()

    timezone_str = tf.timezone_at(lng=float(df.iloc[0]['lon']), lat=float(df.iloc[0]['lat']))  # Get timezone string
    return timezone_str

#----------------------------------------------------------------------------------------
def localize_forecast_dataframe(df):
    '''
    Takes a Open Weather forecast dataframe and converts
    the predict_day and forecat_time to localaize based on lat and lon
    
    Parmameters
    --------------------------------
    df : pandas dataframe object
         The Open Weather CSV converted to a data frame object
         
    Returns
    ----------------------------------
    df :  pandas dataframe object
         The Open Weather data frame object with localized tiemstamp columns
    '''
    #get timezone
    timezone_str =get_local_timezone_from_forecast_df(df)
    if timezone_str is not None:
        # Convert the datetime to the local timezone
        local_timezone = pytz.timezone(timezone_str)
        # Localize 'predict_day' and 'forecast_time'
        print (datetime.now().isoformat(sep=' ') + '     Localization of predict time column')   
        df['predict_time'] = df['predict_time'].dt.tz_convert(local_timezone)
        print (datetime.now().isoformat(sep=' ') + '    Localization of forecast  time column')   
        df['forecast_time'] = df['forecast_time'].dt.tz_convert(local_timezone)
        print (datetime.now().isoformat(sep=' ') + '   All localizations complete')   
    
        # Redo these columns after localizations
        df['forecast_day'] = df['forecast_time'].dt.date    
        df["predict_day"] = df["predict_time"].dt.date
        
        # Calculate the difference in days between foreast and predict
        df["day_diff"] = (df["forecast_day"] - df["predict_day"]).apply(lambda x: x.days)
        #calculate differenc in hours between forrecast and predict
        df['hour_diff'] = ((df['forecast_time'] - df['predict_time']) / np.timedelta64(1, 'h')).round().astype(int)
    
    return df


#----------------------------------------------------------------------------------------
def flatten_column_names(cols):
    '''
    Flatten a MultiIndex column by:
    - If level_1 is in ('mean', 'min', 'max', 'sum' etc.), append it to the base name.
    - If level_1 is 'first' or 'last' (or ''), drop it from the column name.
    
    Parmameters
    --------------------------------
    cols : list
         list of the current columns (both levels) from data frame
         
    Returns
    ----------------------------------
    flattened :  list
         new column names that have been merged or have had indicator removed.
    '''
    flattened = []
    for col in cols:
        col_name, agg_func = col[0], col[1]
        if agg_func in ['mean', 'min', 'max', 'sum', 'std', 'median']:
            # e.g. temperature_mean
            flattened.append(f"{col_name}_{agg_func}")
        else:
            # e.g. city or status (dropping 'first'/'last')
            flattened.append(col_name)
    return flattened


#----------------------------------------------------------------------------------------
def circular_mean(deg_values):
    '''
    Compute the circular mean of wind directions in degrees.

    Parmameters
    --------------------------------
    deg_values : list
         list of the degrees
         
    Returns
    ----------------------------------
    mean_angle_deg :  float
          single float (0 to 360).
    '''
    radians = np.deg2rad(deg_values)
    x = np.cos(radians)
    y = np.sin(radians)
    x_mean = x.mean()
    y_mean = y.mean()
    mean_angle_rad = np.arctan2(y_mean, x_mean)
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    if mean_angle_deg < 0:
        mean_angle_deg += 360
    
    return mean_angle_deg

#------------------------------------------------------------------------------------------------------------
def remove_offset_if_match(s):
    '''
    Helper function to strip off the offset from those valid rows
    '''
    m = iso_space_offset_pattern.match(s)
    if m:
        # Rebuild: date + space + time, ignoring group(3)
        return f"{m.group(1)} {m.group(2)}"
    else:
        return s
