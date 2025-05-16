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
import numpy as np
import states

names = {y:x.lower().replace(' ','') for x,y,z,w in states.state_codes}

URL = "https://www.eia.gov/electricity/state/{name}/xls/SEP%20Tables%20for%20{state}.xlsx"

class StateEnergyProfile:
    """StateEnergyProfile implementation"""

    def __init__(self,state:str) -> dict:
        """Access state energy profile


        Arguments:

        * `state`: state USPS code

        * `year`: year (default is latest SEP data)

        Returns:

        * `dict`: SEP data for state and year        
        """
        sales = pd.read_excel(URL.format(name=names[state],state=state),sheet_name="8. Sales",skiprows=2,index_col=0,engine='openpyxl')
        columns = [x.split("\n")[1] for x in sales.columns if x.startswith("Year\n")]
        sales.columns = [x.split("\n")[1] if x.startswith("Year\n") else x for x in sales.columns]

        # rows = ["Residential","Commercial","Industrial","Other","Transportation","Total"]

        category = None
        categories = {
            "Sales": "sales[MWh]",
            "Revenue": "revenue[k$]",
            "Customers": "customers",
            "Average": "prices[$/MWh]",
            }
        rescale = {
            "prices[$/MWh]":(10.0,2),
        }
        self.data = {}
        for n,data in sales.iterrows():
            tag = n.split(" ")[0]
            if tag in categories:
                category = categories[tag]
                self.data[category] = {}
            elif tag in ["Residential","Commercial","Industrial","Other","Transportation","Total"]:
                if tag in self.data[category]:
                    raise RuntimeError(f"tag {tag} is duplicated in {category}")
                self.data[category][tag] = {int(x):(round(y*rescale[category][0],rescale[category][1]) if category in rescale else y) for x,y in data[columns].to_dict().items() if isinstance(y,float) and not np.isnan(y)}

        # self.data = {
        #     "sales[MWh]": [float(x) for x in sales.iloc[1:7][columns]],
        #     "revenue[k$]": [float(x) for x in sales.iloc[8:14][columns]],
        #     "customers": [float(x) for x in sales.iloc[15:21][columns]],
        #     "prices[$/MWh]": [float(x)/100*1000 for x in sales.iloc[22:28][columns]],
        # }

    def __getitem__(self,name):
        return self.data[name]

if __name__ == "__main__":

    assert StateEnergyProfile("AZ")["sales[MWh]"]["Residential"][2023] == 38992365.0, "get_state('AZ') failed"
    assert StateEnergyProfile("CA")["sales[MWh]"]["Residential"][2023] == 82820898.0, "get_state('CA') failed"