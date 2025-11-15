"""Test the wecc240() model conversion from PSS/E to PyPower"""

from psse2pp import PSSE2PP
from wecc240 import wecc240

#
# Verify the WECC 240 model solves correctly
#

# load the model
PSSE2PP.LOADSCALE = 1.0 # global scaling of loads
case = wecc240()

# # print the case data
# import pandas as pd
# pd.options.display.max_columns = None
# pd.options.display.width = None
# pd.options.display.max_rows = None
# from ppmodel import PPModel
# PPModel("wecc240").with_case(case).print(["gencost","dclinecost"])

# solve the model
from pypower.runpf import runpf
from pypower.ppoption import ppoption
result,status = runpf(case,ppoption(VERBOSE=0,OUT_ALL=0))
if status == 0:

    # rerun with verbose output enabled to diagnose failure
    print("ERROR [wecc240]: solver failed")
    runpf(case,ppoption(VERBOSE=3,OUT_ALL=-1)) # redo with lots of output
    exit(1)

# # print results
# from pypower.printpf import printpf
# printpf(result)

print("WECC240 powerflow solved ok.")

import numpy as np
from pypower import idx_bus as bus
bus_i = case["bus"][:,bus.BUS_I].astype(int).astype(str)
vm_err = np.array(1-result["bus"][:,bus.VM])/np.array(case["bus"][:,bus.VM])
va_err = np.array(result["bus"][:,bus.VA]*180/np.pi)-np.array(case["bus"][:,bus.VA])

import matplotlib.pyplot as plt

# plt.figure(figsize=(40,20))

# plt.subplot(2,1,1)
# plt.bar(bus_i,vm_err*100)
# plt.ylabel("VM error (%)")
# plt.xticks(rotation=90)
# plt.grid()

# plt.subplot(2,1,2)
# plt.bar(bus_i,va_err)
# plt.ylabel("VA error (deg)")
# plt.xlabel("Bus ID")
# plt.xticks(rotation=90)
# plt.grid()

# plt.savefig("voltage_error.png")

#
# Plot voltage errors by bus
#
plt.figure(figsize=(40,20))

plt.subplot(2,1,1)
plt.plot(bus_i,case["bus"][:,bus.VM],label="PSS/E")
plt.plot(bus_i,result["bus"][:,bus.VM],label="PyPower")
plt.ylabel("Voltage Magnitude (pu.kV)")
plt.grid()
plt.xticks(rotation=90)
plt.legend()

plt.subplot(2,1,2)
plt.plot(bus_i,case["bus"][:,bus.VA],label="PSS/E")
plt.plot(bus_i,result["bus"][:,bus.VA]*180/np.pi,label="PyPower")
plt.ylabel("Voltage Angle (deg)")
plt.xlabel("Bus ID")
plt.grid()
plt.xticks(rotation=90)
plt.legend()

plt.savefig("voltage.png")

plt.close()

#
# Plot voltage errors in order
#
import pandas as pd
errors = pd.DataFrame({
    "vm_err":np.abs(vm_err),
    "va_err":np.abs(va_err),
    "bus_i":bus_i}).set_index("bus_i")

plt.figure(figsize=(20,10))

ax = plt.subplot(1,2,1)
(errors[["vm_err"]]*100)\
    .sort_values("vm_err",ascending=False)\
    .plot(grid=True,
        ylabel="Voltage Magnitude Error (%)",
        ax=ax,
        legend=False,
        xlabel="Bus rank")

ax = plt.subplot(1,2,2)
errors[["va_err"]]\
    .sort_values("va_err",ascending=False)\
    .plot(grid=True,
        ylabel="Voltage Angle Error (deg)",
        ax=ax,
        legend=False,
        xlabel="Bus rank")

plt.savefig("voltage_errors.png")
