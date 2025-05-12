"""EIA state energy profile accessor

To access a state energy profile:

~~~
>>> import eia
>>> sep = eia.StateEnergyProfile("CA")
>>> sep["sales[MWh]"]["2023"]["Residential"]
82820898.0
~~~

Sectors available in SEP:

* `Residential`
* `Commercial`
* `Industrial`
* `Other`
* `Transportation`
* `Total`
"""
import os
import sys
import pandas as pd

URL = "https://www.eia.gov/electricity/state/california/xls/SEP%20Tables%20for%20{state}.xlsx"

class StateEnergyProfile:
    """StateEnergyProfile implementation"""

    def __init__(self,state):
        """Access state energy profile

        Arguments:

        * `state`: state USPS code
        """
        sales = pd.read_excel(URL.format(state=state),sheet_name="8. Sales",skiprows=2,index_col=0)
        columns = [x.split("\n")[1] for x in sales.columns if x.startswith("Year\n")]
        sales.columns = [x.split("\n")[1] if x.startswith("Year\n") else x for x in sales.columns]

        rows = ["Residential","Commercial","Industrial","Other","Transportation","Total"]
        self.data = {
            "sales[MWh]": sales.iloc[1:7][columns],
            "revenue[k$]": sales.iloc[8:14][columns],
            "customers": sales.iloc[15:21][columns],
            "prices[$/MWh]": sales.iloc[22:28][columns]/100*1000,
        }

    def __getitem__(self,name):
        return self.data[name]

if __name__ == "__main__":

    assert StateEnergyProfile("CA")["sales[MWh]"]["2023"]["Residential"] == 82820898.0, "get_state('CA') failed"