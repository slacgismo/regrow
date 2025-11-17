"""Test the wecc240() model conversion from PSS/E to PyPower"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psse2pp import PSSE2PP
from wecc240 import wecc240
from ppmodel import PPModel
from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.ppoption import ppoption
from pypower import idx_bus as bus

os.makedirs("tests",exist_ok=True)

errors = 0

def plot(basecase:dict,
    testcase:dict,
    prefix:str="",
    ):
    """Generate test result plots

    Arguments:

    basecase: the reference case to use for plotting errors

    testcase: the test case to use for plotting errors

    prefix: plot file prefix to use for saving plots
    """

    # pylint: disable=too-many-instance-attributes

    bus_i = basecase["bus"][:,bus.BUS_I].astype(int).astype(str)
    vm_err = np.array(1-testcase["bus"][:,bus.VM])/np.array(basecase["bus"][:,bus.VM])
    va_err = np.array(testcase["bus"][:,bus.VA])-np.array(basecase["bus"][:,bus.VA])

    #
    # Plot voltage errors by bus
    #
    plt.figure(figsize=(44,28))

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

    plt.suptitle(f"{os.path.basename(prefix).replace('_',' ').title()} Voltage")
    plt.savefig(f"{prefix}voltage.png")

    plt.close()

    #
    # Plot voltage errors in order
    #
    data = pd.DataFrame({
        "vm_err":np.abs(vm_err),
        "va_err":np.abs(va_err),
        "bus_i":bus_i}).set_index("bus_i")

    plt.figure(figsize=(20,10))

    ax = plt.subplot(1,2,1)
    (data[["vm_err"]]*100)\
        .sort_values("vm_err",ascending=False)\
        .plot(grid=True,
            ylabel="Voltage Magnitude Error (%)",
            ax=ax,
            legend=False,
            xlabel="Bus rank")

    ax = plt.subplot(1,2,2)
    data[["va_err"]]\
        .sort_values("va_err",ascending=False)\
        .plot(grid=True,
            ylabel="Voltage Angle Error (deg)",
            ax=ax,
            legend=False,
            xlabel="Bus rank")

    plt.suptitle(f"{os.path.basename(prefix).replace('_',' ').title()} Voltage Errors")
    plt.savefig(f"{prefix}voltage_errors.png")

    plt.close()


#
# Verify the original WECC 240 model
#

# load the model
PSSE2PP.LOADSCALE = 1.0 # global scaling of loads
original = wecc240()

# save the case data
# PPModel("wecc240").set_case(case).print(["gencost","dclinecost"])
with open("tests/wecc240_original.py","w",encoding="utf-8") as fh:
    PPModel("wecc240").set_case(original).save_case(fh)

    # solve the original powerflow from PSSE
    original_solution,status = runpf(original,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        print(f"ERROR [wecc240]: original case powerflow failed (see {fh.name})")
        errors += 1
    else:
        print("Original WECC240 powerflow solved ok.",flush=True)

    print("Saving comparison plots to tests folder",end="...",flush=True)
    plot(basecase=original,testcase=original_solution,prefix="tests/original_")
    plot(basecase=original,testcase=original_solution,prefix="tests/original_")
    print("done")

    # solve the original model DCOPF
    dcopf = rundcopf(original,ppoption(VERBOSE=0,OUT_ALL=0))
    if not dcopf["success"]:
        print(f"ERROR [wecc240]: original case dcopf failed (see {fh.name})")
        errors += 1
    else:
        print("Original WECC240 DC OPF solved ok.",flush=True)

# solve the DCOPF powerflow
with open("tests/wecc240_original_dcopf.py","w",encoding="utf-8") as fh:
    PPModel("wecc240").set_case(dcopf).save_case(fh)
    dcopf_solution,status = runpf(dcopf,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        print(f"ERROR [wecc240]: original case dcopf powerflow failed (see {fh.name})")
        errors += 1
    else:
        print("WECC240 DC OPF powerflow solved ok.",flush=True)

    print("Saving comparison plots to tests folder",end="...",flush=True)
    plot(basecase=original,testcase=dcopf_solution,prefix="tests/original_dcopf_")
    plot(basecase=original,testcase=dcopf_solution,prefix="tests/original_dcopf_")
    print("done")


#
# Verify the scheduling WECC 240 model
#
scheduling = wecc240(options=["SCHEDULING"])
with open("tests/wecc240_scheduling.py","w",encoding="utf-8") as fh:
    PPModel("wecc240").set_case(original).save_case(fh)

    # solve the schedulig powerflow from PSSE
    scheduling_solution,status = runpf(scheduling,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        print(f"ERROR [wecc240]: scheduling case powerflow failed (see {fh.name})")
        errors += 1
    else:
        print("Scheduling WECC240 powerflow solved ok.",flush=True)

    # solve the schedule model DCOPF
    dcopf = rundcopf(scheduling,ppoption(VERBOSE=0,OUT_ALL=0))
    if not dcopf["success"]:
        print(f"ERROR [wecc240]: scheduling case dcopf failed (see {fh.name})")
        errors += 1
    else:
        print("Schedule WECC240 DC OPF solved ok.",flush=True)
        

if errors > 0:
    print(f"WECC240 failed {errors} test.")

sys.exit(errors)
