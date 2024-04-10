"""Generate commercial building loadshape plots"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("loadshapes",exist_ok=True)

for file in sorted(os.listdir("com_loadshapes")):
    if file.startswith("G") and file.endswith(".csv.zip"):
        print("Processing",file,end="...",flush=True,file=sys.stderr)
        data = pd.read_csv(os.path.join("com_loadshapes",file),index_col=["building_type","datetime"])
        for building_type in data.index.get_level_values(0).unique():
            png = os.path.join("loadshapes",file.replace(".csv.zip",f"-{building_type}.png"))
            if not os.path.exists(png):
                data.loc[building_type].plot(figsize=(20,8))
                plt.savefig(png)
                plt.close()
        print("done",file=sys.stderr)
