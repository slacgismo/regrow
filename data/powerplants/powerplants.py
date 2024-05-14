import pandas as pd
import math

pd.options.display.max_columns = None
pd.options.display.width = None

plants = pd.read_csv("nuclear.csv")
print(plants)

with open("nuclear.glm","w") as glm:
    for _,data in plants.iterrows():
        print(f"""object powerplant
{{
    name "{data['name']}_{1 if math.isnan(data['unit_id']) else int(data['unit_id'])}";
    // type "{data['type']}";
    // model "{data['model']}";
    // county "{data['county']}";
    // state "{data['state']}";
    latitude {data['latitude']};
    longitude {data['longitude']};
    in_svc "{max(2000,int(data['start[y]']))}-01-01 00:00:00 UTC";
    out_svc {'NEVER' if math.isnan(data['retirement[y]']) else "+"+str(int(data['retirement[y]']))+"-01-01 00:00:00 UTC"};
    capacity "{data['capacity[MW]']} MW";
}}
""")
