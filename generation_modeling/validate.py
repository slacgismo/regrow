"""Validate the generation data and cost model"""

from pypower.runopf import runopf
from wecc240 import wecc240
import numpy as np
import pandas as pd

# load the WECC model
data = wecc240()

# load gen data and gen cost files
gen = pd.read_csv("gen.csv").to_numpy()
gencost = pd.read_csv("gencost.csv").to_numpy()

# convert busname to busid
busmap = pd.read_csv("bus_data.csv",index_col=['busname'])
gen[:,0] = [float(busmap.loc[x].busID) for x in gen[:,0]]

# replace default gen/gencost arrays in model
data["gen"] = gen
data["gencost"] = gencost

# run full AC OPF
runopf(data)
