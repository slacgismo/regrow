import requests
import pandas as pd
import time

api_key = "YOUR API KEY"

plant_ids = [*range(50000, 70000)]
master_df_list = list()

for plant_id in plant_ids:
    tries= 0
    while tries < 10:
        try:
            response = requests.get(
                "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/?frequency=monthly&data[0]=county&data[1]=latitude&data[2]=longitude&data[3]=nameplate-capacity-mw&data[4]=operating-year-month&data[5]=planned-retirement-year-month&facets[plantid][]=" + str(plant_id) + "&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=200",
                params={'api_key': api_key}).json()
            df = pd.DataFrame(response['response']['data'])
            if len(df) == 0:
                break
            else:
                df['period'] = pd.to_datetime(df['period'])
                df['max_date_logged'] = df.groupby(['plantid',
                                                    'generatorid',
                                                    'energy_source_code'])["period"].transform("max")
                df = df[df['period'] == df['max_date_logged']]
                master_df_list.append(df)
                master_df = pd.concat(master_df_list)
                master_df.to_csv("eia_plant_generators.csv", index=False)
            break
        except:
            tries += 1

master_df = pd.read_csv("eia_plant_generators.csv")
# For each specific plant ID, pull down all of its 923 data going back to 
# start

plant_ids = list(master_df['plantid'].drop_duplicates())

for plant_id in plant_ids:
    tries= 0
    while tries < 10:
        try:
            response = requests.get(
                "https://api.eia.gov/v2/electricity/facility-fuel/data/?frequency=monthly&data[0]=consumption-for-eg&data[1]=generation&data[2]=gross-generation&data[3]=total-consumption&facets[plantCode][]=" + str(plant_id) + "&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000",
                params={'api_key': api_key}).json()
            df = pd.DataFrame(response['response']['data'])
            df.to_csv("./eia_plant_data/" + str(plant_id) + ".csv", index=False)
            break
        except:
            tries += 1
