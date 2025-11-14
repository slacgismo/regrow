"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from wecc240 import wecc240
    case = wecc240()

    from pypower.runpf import runpf
    runpf(case)
"""

from psse import PSSE
from psse2pp import PSSE2PP

def wecc240():
    """Load and convert the WECC 240 PSSE RAW model to a PyPOWER case"""

    psse = PSSE("wecc240")
    return PSSE2PP(psse).model.case

if __name__ == "__main__":

    #
    # Verify the WECC 240 model solves correctly
    #

    # load the model
    # PSSE2PP.LOADSCALE = 1.0 # global scaling of loads
    case = wecc240()

    # print case data
    # import numpy as np
    # np.set_printoptions(
    #     precision=4,
    #     threshold=10000,
    #     edgeitems=5,
    #     linewidth=10000,
    #     formatter={"float":lambda x:f"{x:10.4g}"})
    # print(case)

    # solve the model
    from pypower.runpf import runpf
    from pypower.ppoption import ppoption
    result,status = runpf(case,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        print("ERROR [wecc240]: solver failed")
        runpf(case,ppoption(VERBOSE=3,OUT_ALL=-1)) # redo with lots of output
        exit(1)

    # output results
    from pypower.printpf import printpf
    printpf(result)
