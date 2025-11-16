"""Test the wecc240() model conversion from PSS/E to PyPower"""

import os
import numpy as np
import matplotlib.pyplot as plt
from psse2pp import PSSE2PP
from wecc240 import wecc240
from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.runopf import runopf as runacopf
from pypower.ppoption import ppoption
from pypower import idx_bus as bus

os.makedirs("tests",exist_ok=True)

errors = 0

def plot(basecase:dict,
    testcase:dict,
    prefix:str="",
    legend=None):
    """Generate test result plots

    Arguments:

    basecase: the reference case to use for plotting errors

    testcase: the test case to use for plotting errors

    prefix: plot file prefix to use for saving plots
    """

    bus_i = basecase["bus"][:,bus.BUS_I].astype(int).astype(str)
    vm_err = np.array(1-testcase["bus"][:,bus.VM])/np.array(basecase["bus"][:,bus.VM])
    va_err = np.array(testcase["bus"][:,bus.VA])-np.array(basecase["bus"][:,bus.VA])

    #
    # Plot voltage errors by bus
    #
    plt.figure(figsize=(40,20))

    plt.subplot(2,1,1)
    plt.plot(bus_i,basecase["bus"][:,bus.VM],label="Base case")
    plt.plot(bus_i,testcase["bus"][:,bus.VM],label="Test case")
    plt.ylabel("Voltage Magnitude (pu.kV)")
    plt.grid()
    plt.xticks(rotation=90)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(bus_i,basecase["bus"][:,bus.VA],label="Base case")
    plt.plot(bus_i,testcase["bus"][:,bus.VA]*180/np.pi,label="Test case")
    plt.ylabel("Voltage Angle (deg)")
    plt.xlabel("Bus ID")
    plt.grid()
    plt.xticks(rotation=90)
    plt.legend()

    plt.savefig(f"{prefix}voltage.png")

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

    plt.savefig(f"{prefix}voltage_errors.png")

#
# Verify the WECC 240 model solves correctly
#

# load the model
PSSE2PP.LOADSCALE = 1.0 # global scaling of loads
original = wecc240()

# save the case data
from ppmodel import PPModel
# PPModel("wecc240").set_case(case).print(["gencost","dclinecost"])
PPModel("wecc240").set_case(original).save_case(open("tests/wecc240_original.py","w"))

# solve the original powerflow from PSSE
original_solution,status = runpf(original,ppoption(VERBOSE=0,OUT_ALL=0))
if status == 0:
    print("ERROR [wecc240]: original case powerflow failed (see wecc240_original.py)")
    errors += 1

# solve the original model DCOPF
dcopf = rundcopf(original,ppoption(VERBOSE=0,OUT_ALL=0))
if not dcopf["success"]:
    print("ERROR [wecc240]: original case dcopf failed (see wecc240_original.py)")
    errors += 1

# solve the DCOPF powerflow
PPModel("wecc240").set_case(dcopf).save_case(open("tests/wecc240_original_dcopf.py","w"))
dcopf_solution,status = runpf(dcopf,ppoption(VERBOSE=0,OUT_ALL=0))
if status == 0:
    print("ERROR [wecc240]: original case dcopf powerflow failed (see wecc240_original_dcopf.py)")
    errors += 1

if errors == 0:
    print("WECC240 powerflow solved ok.")
else:
    print(f"WECC240 failed {errors} test.")

plot(basecase=original,testcase=original_solution,prefix="tests/original_")
plot(basecase=original,testcase=original_solution,prefix="tests/original_")

plot(basecase=original,testcase=dcopf_solution,prefix="tests/original_dcopf_")
plot(basecase=original,testcase=dcopf_solution,prefix="tests/original_dcopf_")

exit(errors)