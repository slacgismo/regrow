"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from pypower.runpf import runpf
    from wecc240 import wecc240
    case = wecc240()
    runpf(case)
"""

import pandas as pd

from psse import PSSE
from psse2pp import PSSE2PP
from ppmodel import PPModel

def wecc240():
    """Load and convert the WECC 240 PSSE model to a PyPOWER case"""

    psse = PSSE("wecc240")
    return PSSE2PP(psse).model.case

if __name__ == "__main__":

    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = None

    # PSSE.VERBOSE = True
    # PSSE.DEBUG = True

    # PPModel.VERBOSE = True

    # PSSE2PP.DEBUG = True

    from pypower.runpf import runpf
    case = wecc240()
    print(case)
    # runpf(case)