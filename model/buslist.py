"""This script extracts the powerplants_aggregated.csv file and generates the
buslist for each type of generator in the WECC model
"""
import pandas as pd

pd.options.display.max_rows = None

data = pd.read_csv("powerplants_aggregated.csv",index_col=0)

for gen in data.gen.unique():
    data[data.gen==gen].to_csv(f"buslist_{gen}.csv")
