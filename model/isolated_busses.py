import pandas as pd

nolink = [  5,  11,  22,  31,  47,  74,  76,  77,  86,  93,  99, 106, 110,
       119, 120, 155, 156, 169, 170, 184, 189, 195, 196, 201, 214, 220,
       235, 242]

busses = pd.read_csv("wecc240_psse_bus.csv")

print(busses.loc[nolink])