"""Load modeling library

The LoadModel class implements an electric load model estimator/predictor.
The independent variables are date/time, temperature, and or price.

Reading Data
------------

Load data can be imported using the read_{csv,xlsx}() methods. 


"""

# TODO: parameterization


# mods
# 1. gridlabd uses hour beginning instead of hour-ending (remove +1hr)

# suggestions
# 1. calculate HI from DB & RH and use that for temperature
# 2. add slope to Hs array in both fit and predict
# 3. convert to sklearn fit_predict() implementation

import os
import marimo as mo
import pandas as pd
import numpy as np
import cvxpy as cvx
import datetime as dt
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import r2_score
import statsmodels.api as sm
from spcqe import make_basis_matrix, make_regularization_matrix

pd.options.display.max_columns=None
pd.options.display.width=None

class LoadModel:

    def __init__(self,name="unnamed"):

        self.name = name
        self.data = None

    def read_xlsx(self,pathname:str,
            sheetname:str,
            /,
            time_col:int|str|list[int|str]=0, # time column(s) to read (joined)
            load_col:int|str|list[int|str]=1, # load column(s) to read (summed)
            temperature_col:int|str|list=2, # temperature column to read
            price_col:int|str|list=None, # price column to read
            datetime:str="dt", # datetime column name
            load:str|None="Q", # load column name, if any
            temperature:str|None="T", # temperature column name, if any
            price:str|None="P", # price column name, if any
            timezone:None|str|dt.tzinfo=None, # timezone localization
            index_split:str|list[str]|None=None, # e.g., year, season, month, dayofweek, hour
            ordinal_hours:bool=False, # hours as 1-24 instead of 0-23
            hour_ending:bool=False, # hours ending instead of hours starting
            keep_columns:bool|list|None=None, # original columns to keep (True=all)
            index:str|list[str]|None=None,
            converters=None, # dict of callables on resulting columns
            inplace=False, # save to self.date
            ):
        """Read XLSX file

        Reads an Excel spreadsheet and processes data into standard load model
        form.

        Parameters
        ----------

        pathname: XLSX path and file name.

        sheetname: XLSX worksheet name.

        time_col: Number, name, or list of numbers or names containing input 
                  time data. Two columns can be used to specify date and time
                  separately, in which case they will be combined to form the
                  time column.

        load_col: Number, name, or list of numbers or names containing
                  required input load data. Multiple columns are summed to form
                  the load column.

        temperature_col: Number or name of requirement input temperature
                         column. 

        price_col: Number or name of optional input price column.

        datetime: Name of output date/time index.

        timezone: Timezone localization (see DatetimeIndex.tz_localize)

        load: Name of output load column.

        temperature: Name of output temperature column.

        price: Name of output price column, if any.

        index_split: Additional columns formed from attributes of index
                     (see pandas.DatetimeIndex attributes)

        ordinal_hours: Specify when hours are ordinal, i.e., 1-24, instead of
                       0-23

        keep_columns: Specify which input columns to preserve in output

        index: column name to use as index or callable to create index

        converters: converters to apply to output columns

        inplace: append/replace/join data to model data (True='replace')

        Attributes
        ----------

        name: name of the model

        data: model data

        Returns
        -------

        pd.DataFrame: resulting data
        """

        # setup columns to read
        columns = list(time_col) if isinstance(time_col,list) else [time_col]
        columns += load_col if isinstance(load_col,list) else [load_col]
        columns += [temperature_col]
        if not price_col is None:
            columns += [price_col]
        df = pd.read_excel(pathname,sheet_name=sheetname,usecols=columns)

        def drop(cols:str|list[str]):
            """Drop columns not in keep_columns"""
            if keep_columns == True:
                return
            for col in cols if isinstance(cols,list) else [cols]:
                if keep_columns is None or not col in keep_columns:
                    df.drop(col,axis=1,inplace=True)
        def aslist(item):
            return item if isinstance(item,list) else ( [] if item is None else [item])

        # convert column numbers to column names
        if isinstance(time_col,list):
            time_col = [df.columns[x] if isinstance(x,int) else x for x in time_col]
        elif isinstance(time_col,int):
            time_col = df.columns[time_col]
        if isinstance(load_col,list):
            load_col = [df.columns[x] if isinstance(x,int) else x for x in load_col]
        elif isinstance(load_col,int):
            load_col = df.columns[load_col]
        if isinstance(temperature_col,int):
            temperature_col = df.columns[temperature_col]
        if isinstance(price_col,int):
            price_col = df.columns[price_col]

        # collate time_col
        if isinstance(time_col,(int,str)):
            df[datetime] = pd.to_datetime(dt[time_col])
        elif isinstance(time_col,list):
            date = df[time_col[0]].astype(str)
            hour = (df[time_col[1]]-1 if ordinal_hours else df[time_col[1]]).astype(int)
            dt = [" ".join(dt)+":00:00" for dt in zip(date,[f"{x:02d}" for x in hour])]
            df[datetime] = pd.to_datetime(dt).tz_localize(timezone)
        if hour_ending:
            df[datetime] -= dt.timedelta(hour=1)

        drop(time_col)

        # collate loads
        if isinstance(load,(str,list)):
            df[load] = df[load_col].sum(axis=1) if isinstance(load_col,list) else df[load_col] 
            drop(load_col)
        elif not load is None:
            raise TypeError(f"{load=} has invalid type {repr(type(temperature))}")

        if isinstance(temperature,str):
            df[temperature] = df[temperature_col]
            drop(temperature_col)
        elif not temperature is None:
            raise TypeError(f"{temperature=} has invalid type {repr(type(temperature))})")

        # run converters
        if converters is None:
            converters = {}
        if not isinstance(converters,dict):
            raise TypeError(f"converters is not a dict")
        for column,converter in converters.items():
            if not column in [datetime,temperature,load,price]:
                raise KeyError(f"converter {column=} is not found")
            if column in df.columns:
                df[column] = df[column].map(converter)

        # set index
        if isinstance(index,(int,str,list)):
            df.set_index(index,inplace=True)
        elif callable(index):
            df.index = index(df)
            df.sort_index(inplace=True)
        elif not index is None:
            raise ValueError(f"{index=} is not valid")

        # run index splits
        for ds in aslist(index_split):
            if hasattr(df.index,ds):
                df[ds] = getattr(df.index,ds)
            else:
                raise ValueError(f"index_split '{ds}' is not valid")

        if inplace in [True,"replace"] or self.data is None:
            self.data = df
        elif inplace == "append":
            self.data = pd.concat([self.data,df]).sort_index()
        elif inplace == "join":
            self.data = self.data.join(df)
        elif not inplace in [False,None]:
            raise ValueError(f"{inplace=} is invalid")

        return df

    def to_csv(self,*args,**kwargs):
        self.data.to_csv(*args,**kwargs)

    def resample(self,interval,inplace=False):
        return

if __name__ == "__main__":
    model = LoadModel()
    for year in range(2020,2023):
        model.read_xlsx(f"NE_ISO_Data/{year}_smd_hourly.xlsx","ME",
            time_col=["Date","Hr_End"],
            load_col="RT_Demand",
            temperature_col="Dry_Bulb",
            temperature="T[degC]",
            load="P[GW]",
            ordinal_hours=True,
            index="dt",
            timezone="EST",
            converters={
                "P[GW]": lambda x: round(x/1000,3),
                "T[degC]": lambda x: round((x-32)*5/9,1)
                },
            index_split=["year","quarter","month","day","weekday","hour","minute","dayofyear"],
            inplace='append',
            )
    print(model.data)

