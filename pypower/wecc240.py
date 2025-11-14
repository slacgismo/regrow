"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from wecc240 import wecc240
    case = wecc240()

    from pypower.runpf import runpf
    runpf(case)
"""

import pandas as pd

from psse import PSSE
from psse2pp import PSSE2PP

def wecc240():
    """Load and convert the WECC 240 PSSE RAW model to a PyPOWER case"""

    psse = PSSE("wecc240")
    return PSSE2PP(psse).model.case

if __name__ == "__main__":

    from pypower.printpf import printpf
    from pypower.runpf import runpf

    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = None

    case = wecc240()
    result = runpf(case)
    printpf(result)
