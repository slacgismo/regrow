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
from ppmodel import PPModel

def wecc240() -> dict:
    """Load and convert the WECC 240 PSSE RAW model to a PyPOWER case

    Returns:

    dict: pypower case (see PyPower documentation)
    """

    raw = PSSE("wecc240")
    model = PSSE2PP(raw).model
    return model.case

if __name__ == "__main__":

    #
    # Verify the WECC 240 model solves correctly
    #

    # load the model
    # PSSE2PP.LOADSCALE = 1.0 # global scaling of loads
    case = wecc240()

    # print the case data
    import pandas as pd
    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = None
    PPModel("wecc240").with_case(case).print(["dcline"])

    # solve the model
    from pypower.runpf import runpf
    from pypower.ppoption import ppoption
    result,status = runpf(case,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        print("ERROR [wecc240]: solver failed")
        runpf(case,ppoption(VERBOSE=3,OUT_ALL=-1)) # redo with lots of output
        exit(1)

    # print results
    # from pypower.printpf import printpf
    # printpf(result)
