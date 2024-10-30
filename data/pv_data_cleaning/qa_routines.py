"""
QA routines for power, irradiance, temperature, and wind speed. These
filtering methods are applied to data with processed data as an output.
"""

import pvanalytics
from rdtools import filtering
import pvlib
import pandas as pd
from pvanalytics.quality import data_shifts as ds
from pvanalytics.quality import gaps
from pvanalytics.quality.outliers import zscore
from pvanalytics.features.daytime import power_or_irradiance
from pvanalytics.quality.time import shifts_ruptures
from pvanalytics.features import daytime
from statistics import mode
import ruptures as rpt
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

def run_power_stream_routine(power_time_series, latitude,
                             longitude, psm3, mount):
    """
    Function stringing all the PVAnalytics functions together to run power QA.
    """
    # Get the time frequency of the time series
    power_time_series.index = pd.to_datetime(power_time_series.index, utc=True)
    freq_minutes = mode(
        power_time_series.index.to_series().diff(
        ).dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    power_time_series = power_time_series.asfreq(data_freq)
    
    # BASIC DATA CHECKS (STALE, OUTLIERS)
    
    # REMOVE STALE DATA (that isn't during nighttime periods or clipped)
    # Day/night mask
    daytime_mask = power_or_irradiance(power_time_series)
    # Clipped data (uses rdtools filters)
    clipping_mask = filtering.xgboost_clip_filter(
        power_time_series)
    stale_data_mask = gaps.stale_values_round(
        power_time_series,
        window=3,
        decimals=2)
    
    stale_data_mask = (stale_data_mask & daytime_mask
                       & clipping_mask)
    # REMOVE NEGATIVE DATA
    negative_mask = (power_time_series < 0)
    # FIND ABNORMAL PERIODS
    daily_min = power_time_series.resample('D').min()   
    series_min = 0.1 * power_time_series.mean()
    erroneous_mask = (daily_min >= series_min)
    erroneous_mask = erroneous_mask.reindex(
        index=power_time_series.index,
        method='ffill',
        fill_value=False)
    # FIND OUTLIERS (Z-SCORE FILTER)
    zscore_outlier_mask = zscore(power_time_series,
                                 zmax=4,
                                 nan_policy='omit')
    # Filter the time series, taking out all of the issues
    issue_mask = ((~stale_data_mask) & (~negative_mask) &
              (~erroneous_mask) & (~zscore_outlier_mask))
    
    power_time_series = power_time_series[issue_mask].copy()
    power_time_series = power_time_series.asfreq(data_freq)
    
    # DATA COMPLETENESS CHECK
    
    # Visualize daily data completeness
    power_time_series = power_time_series.asfreq(data_freq)
    daytime_mask = power_or_irradiance(power_time_series)
    power_time_series.loc[~daytime_mask] = 0
    # Trim the series based on daily completeness score
    trim_series_mask = \
    pvanalytics.quality.gaps.trim_incomplete(
        power_time_series,
        minimum_completeness=.25,
        freq=data_freq)
    
    power_time_series = power_time_series[trim_series_mask]
    
    # TIME SHIFT DETECTION
    if ((len(power_time_series.resample('D').mean().dropna()) >=150) &
            (len(power_time_series.drop_duplicates()) > 1000)):
        # Get the modeled sunrise and sunset time series based
        # on the system's latitude-longitude coordinates
        modeled_sunrise_sunset_df = \
        pvlib.solarposition.sun_rise_set_transit_spa(
             power_time_series.index, latitude, longitude)
        
        # Calculate the midday point between sunrise and sunset
        # for each day in the modeled power series
        modeled_midday_series = \
        modeled_sunrise_sunset_df['sunrise'] + \
            (modeled_sunrise_sunset_df['sunset'] -
             modeled_sunrise_sunset_df['sunrise']) / 2
        
        # Run day-night mask on the power time series
        daytime_mask = power_or_irradiance(
            power_time_series, freq=data_freq,
            low_value_threshold=.005)
        
        # Generate the sunrise, sunset, and halfway points
        # for the data stream
        sunrise_series = daytime.get_sunrise(daytime_mask)
        sunset_series = daytime.get_sunset(daytime_mask)
        midday_series = sunrise_series + ((sunset_series -
                                           sunrise_series)/2)
        
        # Convert the midday and modeled midday series to daily
        # values
        midday_series_daily, modeled_midday_series_daily = (
            midday_series.resample('D').mean(),
            modeled_midday_series.resample('D').mean())
        
        # Set midday value series as minutes since midnight,
        # from midday datetime values
        midday_series_daily = (
            midday_series_daily.dt.hour * 60 +
            midday_series_daily.dt.minute +
            midday_series_daily.dt.second / 60)
        modeled_midday_series_daily = \
            (modeled_midday_series_daily.dt.hour * 60 +
             modeled_midday_series_daily.dt.minute +
             modeled_midday_series_daily.dt.second / 60)
        
        # Estimate the time shifts by comparing the modelled
        # midday point to the measured midday point.
        is_shifted, time_shift_series = shifts_ruptures(
            midday_series_daily, modeled_midday_series_daily,
            period_min=15, shift_min=15, zscore_cutoff=1.5)
        
        # Build a list of time shifts for re-indexing. We choose to use dicts.
        time_shift_series.index = pd.to_datetime(
            time_shift_series.index)
        changepoints = (time_shift_series != time_shift_series.shift(1))
        changepoints = changepoints[changepoints].index
        changepoint_amts = pd.Series(time_shift_series.loc[changepoints])
        time_shift_list = list()
        for idx in range(len(changepoint_amts)):
            if changepoint_amts[idx] == 0:
                change_amt = 0
            else:
                change_amt = -1 * changepoint_amts[idx]
            if idx < (len(changepoint_amts) - 1):
                time_shift_list.append({"datetime_start":
                                        str(changepoint_amts.index[idx]),
                                        "datetime_end":
                                            str(changepoint_amts.index[idx + 1]),
                                        "time_shift": change_amt})
            else:
                time_shift_list.append({"datetime_start":
                                        str(changepoint_amts.index[idx]),
                                        "datetime_end":
                                            str(time_shift_series.index.max()),
                                        "time_shift": change_amt})
        
        # Correct any time shifts in the time series
        new_index = pd.Series(power_time_series.index, index=power_time_series.index).dropna()
        for i in time_shift_list:
            if pd.notna(i['time_shift']):
                new_index[(power_time_series.index >= pd.to_datetime(i['datetime_start'])) &
                      (power_time_series.index < pd.to_datetime(i['datetime_end']))] = \
                power_time_series.index + pd.Timedelta(minutes=i['time_shift'])
        power_time_series.index = new_index
        
        # Remove duplicated indices and sort the time series (just in case)
        power_time_series = power_time_series[~power_time_series.index.duplicated(
            keep='first')].sort_index()
                
        # DATA SHIFT DETECTION
        # Set all values in the nighttime mask to 0
        power_time_series = power_time_series.asfreq(data_freq)
        daytime_mask = power_or_irradiance(power_time_series)
        power_time_series.loc[~daytime_mask] = 0
        # Resample the time series to daily mean
        power_time_series_daily = power_time_series.resample(
            'D').mean()
        data_shift_start_date, data_shift_end_date = \
        ds.get_longest_shift_segment_dates(
            power_time_series_daily,
            use_default_models=False,
            method=rpt.Binseg, cost='rbf',
            penalty=15)
            
        power_time_series = power_time_series[
                (power_time_series.index >=
                 data_shift_start_date.tz_convert(
                     power_time_series.index.tz)) &
                (power_time_series.index <=
                 data_shift_end_date.tz_convert(
                     power_time_series.index.tz))]
        
        power_time_series = power_time_series.asfreq(data_freq)
    power_time_series = power_time_series.asfreq(data_freq)
    return power_time_series


def run_irradiance_stream_routine(irradiance_time_series,
                                  latitude, longitude, psm3):
    """
    Function stringing all the PVAnalytics functions together to run irradiance
    QA.
    """
    # Get the time frequency of the time series
    irradiance_time_series.index = pd.to_datetime(irradiance_time_series.index, utc=True)
    freq_minutes = mode(
        irradiance_time_series.index.to_series().diff(
        ).dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    irradiance_time_series = irradiance_time_series.asfreq(data_freq)
    
    # BASIC DATA CHECKS (STALE, OUTLIERS)
    
    # REMOVE STALE DATA (that isn't during nighttime periods)
    # Day/night mask
    daytime_mask = power_or_irradiance(irradiance_time_series)
    # Stale data mask
    stale_data_mask = gaps.stale_values_round(irradiance_time_series,
                                              window=3,
                                              decimals=2)
    stale_data_mask = stale_data_mask & daytime_mask
    
    # REMOVE NEGATIVE DATA
    negative_mask = (irradiance_time_series < 0)
    
    # FIND ABNORMAL PERIODS
    daily_min = irradiance_time_series.resample('D').min()
    erroneous_mask = (daily_min > 50)
    erroneous_mask = erroneous_mask.reindex(index=irradiance_time_series.index,
                                            method='ffill',
                                            fill_value=False)
    
    # Remove values greater than or equal to 1300
    out_of_bounds_mask = (irradiance_time_series >= 1300)
    
    # FIND OUTLIERS (Z-SCORE FILTER)
    zscore_outlier_mask = zscore(irradiance_time_series,
                                 zmax=4,
                                 nan_policy='omit')
    # Filter the time series, taking out all of the issues
    issue_mask = ((~stale_data_mask) & (~negative_mask) & (~erroneous_mask) &
                  (~out_of_bounds_mask) & (~zscore_outlier_mask))
    irradiance_time_series = irradiance_time_series[issue_mask].copy()
    
    irradiance_time_series = irradiance_time_series.asfreq(data_freq)
    
    # DATA COMPLETENESS CHECK
    
    irradiance_time_series = irradiance_time_series.asfreq(data_freq)
    daytime_mask = power_or_irradiance(irradiance_time_series)
    irradiance_time_series.loc[~daytime_mask] = 0
    # Trim the series based on daily completeness score
    trim_series_mask = pvanalytics.quality.gaps.trim_incomplete(
        irradiance_time_series,
        minimum_completeness=.25,
        freq=data_freq)
    
    irradiance_time_series = irradiance_time_series[trim_series_mask]
    
    # TIME SHIFT DETECTION
    
    # Get the modeled sunrise and sunset time series based on the system's
    # latitude-longitude coordinates
    modeled_sunrise_sunset_df = pvlib.solarposition.sun_rise_set_transit_spa(
         irradiance_time_series.index, latitude, longitude)
    
    # Calculate the midday point between sunrise and sunset for each day
    # in the modeled irradiance series
    modeled_midday_series = modeled_sunrise_sunset_df['sunrise'] + \
        (modeled_sunrise_sunset_df['sunset'] -
         modeled_sunrise_sunset_df['sunrise']) / 2
    
    # Run day-night mask on the irradiance time series
    daytime_mask = power_or_irradiance(irradiance_time_series,
                                       freq=data_freq,
                                       low_value_threshold=.005)
    
    # Generate the sunrise, sunset, and halfway points for the data stream
    sunrise_series = daytime.get_sunrise(daytime_mask)
    sunset_series = daytime.get_sunset(daytime_mask)
    midday_series = sunrise_series + ((sunset_series - sunrise_series)/2)
    
    # Convert the midday and modeled midday series to daily values
    midday_series_daily, modeled_midday_series_daily = (
        midday_series.resample('D').mean(),
        modeled_midday_series.resample('D').mean())
    
    # Set midday value series as minutes since midnight, from midday datetime
    # values
    midday_series_daily = (midday_series_daily.dt.hour * 60 +
                           midday_series_daily.dt.minute +
                           midday_series_daily.dt.second / 60)
    modeled_midday_series_daily = \
        (modeled_midday_series_daily.dt.hour * 60 +
         modeled_midday_series_daily.dt.minute +
         modeled_midday_series_daily.dt.second / 60)
    
    # Estimate the time shifts by comparing the modelled midday point to the
    # measured midday point.
    is_shifted, time_shift_series = shifts_ruptures(midday_series_daily,
                                                    modeled_midday_series_daily,
                                                    period_min=15,
                                                    shift_min=15,
                                                    zscore_cutoff=1.5)
    # Build a list of time shifts for re-indexing. We choose to use dicts.
    time_shift_series.index = pd.to_datetime(
        time_shift_series.index)
    changepoints = (time_shift_series != time_shift_series.shift(1))
    changepoints = changepoints[changepoints].index
    changepoint_amts = pd.Series(time_shift_series.loc[changepoints])
    time_shift_list = list()
    for idx in range(len(changepoint_amts)):
        if changepoint_amts[idx] == 0:
            change_amt = 0
        else:
            change_amt = -1 * changepoint_amts[idx]
        if idx < (len(changepoint_amts) - 1):
            time_shift_list.append({"datetime_start":
                                    str(changepoint_amts.index[idx]),
                                    "datetime_end":
                                        str(changepoint_amts.index[idx + 1]),
                                    "time_shift": change_amt})
        else:
            time_shift_list.append({"datetime_start":
                                    str(changepoint_amts.index[idx]),
                                    "datetime_end":
                                        str(time_shift_series.index.max()),
                                    "time_shift": change_amt})
        
    # Correct any time shifts in the time series
    new_index = pd.Series(irradiance_time_series.index, index=irradiance_time_series.index).dropna()
    for i in time_shift_list:
        if pd.notna(i['time_shift']):
            new_index[(irradiance_time_series.index >= pd.to_datetime(i['datetime_start'])) &
                  (irradiance_time_series.index < pd.to_datetime(i['datetime_end']))] = \
            irradiance_time_series.index + pd.Timedelta(minutes=i['time_shift'])
    irradiance_time_series.index = new_index
    
    # Remove duplicated indices and sort the time series (just in case)
    irradiance_time_series = irradiance_time_series[~irradiance_time_series.index.duplicated(
        keep='first')].sort_index()
    
    # DATA SHIFT CHECKS
    
    # Set all values in the nighttime mask to 0
    irradiance_time_series = irradiance_time_series.asfreq(data_freq)
    daytime_mask = power_or_irradiance(irradiance_time_series)
    irradiance_time_series.loc[~daytime_mask] = 0
    # Resample the time series to daily mean
    irradiance_time_series_daily = irradiance_time_series.resample('D').mean()
    data_shift_start_date, data_shift_end_date = ds.get_longest_shift_segment_dates(irradiance_time_series_daily,
                                                                                    use_default_models=False,
                                                                                    method=rpt.Binseg, cost='rbf',
                                                                                    penalty=20)
    irradiance_time_series = irradiance_time_series[
            (irradiance_time_series.index >=
             data_shift_start_date.tz_convert(irradiance_time_series.index.tz)) &
            (irradiance_time_series.index <=
             data_shift_end_date.tz_convert(irradiance_time_series.index.tz))]
    
    irradiance_time_series = irradiance_time_series.asfreq(data_freq)
    
    return irradiance_time_series
    


def run_temperature_stream_routine(temp_time_series, data_stream_type):
    """
    Function stringing all the PVAnalytics functions together to run temeprature
    QA.
    """
    # Get the time frequency of the time series
    temp_time_series.index = pd.to_datetime(temp_time_series.index, utc=True)
    freq_minutes = mode(
        temp_time_series.index.to_series().diff(
        ).dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    temp_time_series = temp_time_series.asfreq(data_freq)
    
    # REMOVE STALE DATA
    stale_data_mask = gaps.stale_values_round(temp_time_series,
                                              window=3,
                                              decimals=2, mark = "tail")
    
    # FIND ABNORMAL PERIODS
    temperature_limit_mask = pvanalytics.quality.weather.temperature_limits(
        temp_time_series, limits=(-40, 185))
    temperature_limit_mask = temperature_limit_mask.reindex(
        index=temp_time_series.index,
        method='ffill',
        fill_value=False)
    
    # FIND OUTLIERS (Z-SCORE FILTER)
    zscore_outlier_mask = zscore(temp_time_series,
                                 zmax=4,
                                 nan_policy='omit')
    
    # PERFORM ADDITIONAL CHECKS, INCLUDING CHECKING UNITS (CELSIUS OR FAHRENHEIT)
    temperature_mean = temp_time_series.mean()
    if temperature_mean > 35:
        temp_units = 'F'
    else:
        temp_units = 'C'
    
    print("Estimated Temperature units: " + str(temp_units))
    
    # Run additional checks based on temperature sensor type.
    if data_stream_type == 'module':
        if temp_units == 'C':
            module_limit_mask = (temp_time_series <= 85)
            temperature_limit_mask = (temperature_limit_mask & module_limit_mask)
    if data_stream_type == 'ambient':
        ambient_limit_mask = pvanalytics.quality.weather.temperature_limits(
            temp_time_series, limits=(-40, 120))
        temperature_limit_mask = (temperature_limit_mask & ambient_limit_mask)
        if temp_units == 'C':
            ambient_limit_mask_2 = (temp_time_series <= 50)
            temperature_limit_mask = (temperature_limit_mask &
                                      ambient_limit_mask_2)
    # Filter the time series, taking out all of the issues
    issue_mask = ((~stale_data_mask) & (temperature_limit_mask) &
                  (~zscore_outlier_mask))
    
    temp_time_series = temp_time_series[issue_mask].copy()
    
    # Get the time frequency of the time series
    freq_minutes = mode(temp_time_series.index.to_series().diff().dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    
    temp_time_series = temp_time_series.asfreq(data_freq)
    
    # DATA COMPLETENESS SCORE
        
    # Trim the series based on daily completeness score
    trim_series_mask = pvanalytics.quality.gaps.trim_incomplete(
        temp_time_series,
        minimum_completeness=.25,
        freq=data_freq)
    
    temp_time_series = temp_time_series[trim_series_mask]
    
    # DATA SHIFT CHECKS
    
    # Resample the time series to daily mean
    temp_time_series_daily = temp_time_series.resample('D').mean()
    data_shift_start_date, data_shift_end_date = \
        ds.get_longest_shift_segment_dates(temp_time_series_daily)
    temp_time_series = temp_time_series[(temp_time_series.index >=
                                         data_shift_start_date.tz_convert(temp_time_series.index.tz)) &
                                        (temp_time_series.index <=
                                         data_shift_end_date.tz_convert(temp_time_series.index.tz))]
    return temp_time_series



def run_wind_speed_stream_routine(wind_time_series):
    """
    Function stringing all the PVAnalytics functions together to run temeprature
    QA.
    """
    # Get the time frequency of the time series
    wind_time_series.index = pd.to_datetime(wind_time_series.index, utc=True)
    freq_minutes = mode(
        wind_time_series.index.to_series().diff(
        ).dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    wind_time_series = wind_time_series.asfreq(data_freq)
    
    # REMOVE STALE DATA
    stale_data_mask = gaps.stale_values_round(wind_time_series,
                                              window=3,
                                              decimals=2, mark = "tail")
    
    # FIND OUTLIERS (Z-SCORE FILTER)
    zscore_outlier_mask = zscore(wind_time_series,
                                 zmax=5,
                                 nan_policy='omit')
    
    # Filter the time series, taking out all of the issues
    issue_mask = ((~stale_data_mask) & (~zscore_outlier_mask))
    
    wind_time_series = wind_time_series[issue_mask].copy()
    
    # Get the time frequency of the time series
    freq_minutes = mode(wind_time_series.index.to_series().diff().dt.seconds / 60)
    data_freq = str(freq_minutes) + "min"
    
    wind_time_series = wind_time_series.asfreq(data_freq)
    
    # DATA COMPLETENESS CHECK
    
    # Trim the series based on daily completeness score
    trim_series_mask = pvanalytics.quality.gaps.trim_incomplete(
        wind_time_series,
        minimum_completeness=.25,
        freq=data_freq)
    
    wind_time_series = wind_time_series[trim_series_mask]
    
    # DATA SHIFTS
    
    # Resample the time series to daily mean
    wind_time_series_daily = wind_time_series.resample('D').mean()
    data_shift_start_date, data_shift_end_date = \
        ds.get_longest_shift_segment_dates(wind_time_series_daily)
    wind_time_series = wind_time_series[(wind_time_series.index >=
                                         data_shift_start_date.tz_convert(wind_time_series.index.tz)) &
                                        (wind_time_series.index <=
                                         data_shift_end_date.tz_convert(wind_time_series.index.tz))]
    return wind_time_series


def fit_irradiance_power_stream(power_stream, irrad_stream):
    """
    Get the best-fitting irradiance stream for a particular power stream via
    plotting each stream against each other in a linear regression and taking
    the lowest RMSE.

    Parameters
    ----------
    power_stream : TYPE
        DESCRIPTION.
    irradiance_stream_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rmse_min = 10000
    slope, intercept, r_value, p_value, std_err = linregress(
        np.array(power_stream), np.array(irrad_stream))
    y_pred = intercept + slope * np.array(power_stream)
    rmse = root_mean_squared_error(y_true=np.array(irrad_stream),
                                   y_pred=y_pred)
    return rmse
        