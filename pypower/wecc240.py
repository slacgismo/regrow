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

def wecc240() -> dict:
    """Load and convert the WECC 240 PSSE RAW model to a PyPOWER case

    Returns:

    dict: pypower case (see PyPower documentation)
    """

    raw = PSSE("wecc240")
    model = PSSE2PP(raw).model
    return model.case

