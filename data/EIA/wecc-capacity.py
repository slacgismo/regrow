import pandas as pd
import os
import datetime

NONASCII = {
    "\xe1" : "a",
    "\xe9" : "e",
    "\xed" : "i",
    "\xf1" : "n",
    "\xf3" : "o",
    "\xfc" : "u",
    # may need to add other someday
}

name_to_acronym = {
    "Alabama": "AL",
    "Alaska": "AK", 
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY"
}

def strict_ascii(text):
    return ''.join([NONASCII[x] if x in NONASCII else x for x in text])

def load_wecc_counties(fn='wecc_counties.csv'):
    if os.path.exists(fn):
        return pd.read_csv(fn, index_col=[0], header=0)
    print('Constructing county list from census.gov county info. This could take a couple minutes depending on webiste response...')
    census_url = "https://www2.census.gov/geo/docs/reference/state.txt"
    FIPS_STATES = pd.read_csv(
        census_url,
        delimiter="|",
        index_col=[1],
        usecols=[0,1,2],
        header=0,
        names=["fips","state","name"]
    ).to_dict('index')
    # Counties in WECC
    STATES = ["CA","WA","OR","ID","MT","WY","NV","UT","AZ","NM","CO"] 
    EXCLUDE = [ # Counties to leave out
        # MT
        '30019', # Daniels
        '30021', # Dawson
        '30025', # Fallon
        '30083', # Richland
        '30085', # Roosevelt
        '30091', # Sheridan
        '30109', # Wibaux
        # NM
        '35009', # Curry
        '35025', # Lea
        '35037', # Quay
        '35041', # Roosevelt
        ] 
    INCLUDE = {
        "TX": [
            '141', # El Paso
        ],
        "SD": [
            '033', # Custer
            '047', # Fall River
            '081', # Lawrence
        ]} # counties to add in from other states

    # Assemble county data
    counties_list = []
    for state in (STATES + list(INCLUDE)):
        fips = f"{FIPS_STATES[state]['fips']:02.0f}"

        # County population centroid data
        URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/"
        URL += f"CenPop2020_Mean_CO{fips}.txt"
        df = pd.read_csv(URL,
            converters = {
                "COUNAME": strict_ascii,
                "STATEFP": lambda x: f"{float(x):02.0f}",
                "COUNTYFP": lambda x: f"{float(x):03.0f}",
            },
            usecols = ["STATEFP","COUNTYFP","STNAME","COUNAME","POPULATION","LATITUDE","LONGITUDE"],
            )
        statename = df.STNAME.unique()[0].replace(' ','')
        df.columns = [x.lower() for x in df.columns]
        if state in INCLUDE:
            df.drop(
                df.loc[~df['countyfp'].isin(INCLUDE[state])].index,
                inplace=True,
                axis=0
            )
        rename = {"couname":"name"}
        df.columns = [rename[x] if x in rename else x for x in df.columns]
        df["fips"] = df["statefp"] + df["countyfp"]
        df["state"] = state
        df.drop(["statefp","countyfp"],axis=1,inplace=True)
        df.set_index("fips",inplace=True)
        # df["county"] = df.name + " " + df.state
        df.rename({"name":"county"},axis=1,inplace=True)
        df.drop(["stname","population"],axis=1,inplace=True)
        counties_list.append(df)
    wecc_counties = pd.concat(counties_list).drop(EXCLUDE,axis=0)
    wecc_counties.to_csv(fn)
    return wecc_counties


if __name__ == "__main__":
    # read in the facility and energy type information
    facility_df = pd.read_csv("facility_details.csv")
    energy_type_df = pd.read_csv("energy_types.csv")
    # get state acronyms accordingly
    facility_df['state'] = facility_df['state'].map(name_to_acronym)
    master_df = pd.merge(facility_df[['plant_id', 'plant_name',
                                      'county', 'state', 'latitude', 'longitude', 
                                      'earliest_plant_operation_date',
                                      'source', 'source_id']], energy_type_df, on = 'plant_id')
    
    # get rid of anything that started operating post-2022
    master_df['operating_date'] = pd.to_datetime(master_df['operating_date'])
    master_df = master_df[master_df['operating_date'] < pd.to_datetime('2023-01-01')]
    
    # Get rid of anything that was decommissioned before 2018
    master_df['planned_retirement_date'] = pd.to_datetime(master_df['planned_retirement_date'])
    # set a dummy retirement date for everything still operating
    master_df.loc[master_df['planned_retirement_date'].isna(), 
                  'planned_retirement_date'] = datetime.datetime.now()
    master_df = master_df[~(master_df['planned_retirement_date'] < pd.to_datetime('2018-01-01'))]
    
    # Get a list of all of the WECC counties and filter accordingly
    counties = load_wecc_counties()
    
    wecc_plant_ids = list()
    # Filter all of the sites by county
    for idx, row in counties.iterrows():
        plant_ids = list(master_df[(master_df['county'].str.upper() == row['county'].upper()) &
                                   (master_df['state'].str.upper() == row['state'].upper())]['plant_id'])
        wecc_plant_ids = wecc_plant_ids + plant_ids
    # Now subset the master dataframe to contain all of the WECC generators
    wecc_df = master_df[master_df['plant_id'].isin(wecc_plant_ids)]
    # Build a daily operating capacity summary running between 2018-2022
    date_range = pd.date_range(start="2018-01-01", end="2023-01-01",freq='MS')
    summed_capacity_list = list()
    for date in date_range:
        wecc_df_operating = wecc_df[(wecc_df['operating_date'] < date) &
                                    (wecc_df['planned_retirement_date'] >= date)].drop_duplicates()
        # summ the capacities by generator type
        summed_capacity = wecc_df_operating.groupby('technology')['nameplate_capacity_mw'].sum().reset_index(drop=False)
        summed_capacity['date'] = date
        summed_capacity_list.append(summed_capacity)
    # Concat out the list of dataframes and order by date
    summed_capacity_df = pd.concat(summed_capacity_list)
    summed_capacity_df = summed_capacity_df.sort_values(by='date')
    summed_capacity_pivot = summed_capacity_df.pivot(index='date',
                                                     columns='technology', 
                                                     values='nameplate_capacity_mw')
    summed_capacity_pivot['total_capacity_mw'] = summed_capacity_pivot.sum(axis=1)
    summed_capacity_pivot.to_csv("wecc_capacity_over_time.csv")
    # Pull down the plant level monthly production data
    missing_systems= list()
    master_df_list = list()
    for plant_id in set(wecc_plant_ids):
        file = "./eia-monthly-energy-data/" + str(plant_id) + ".csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df = df[df['common_name'] == 'generation']
            df['measured_on_utc'] = pd.to_datetime(df['measured_on_utc'])
            # Filter to just get data between 2018 and 2022
            df = df[(df['measured_on_utc'].dt.date >= pd.to_datetime('2018-01-01').date()) &
                    (df['measured_on_utc'].dt.date < pd.to_datetime('2023-01-01').date())]
            master_df_list.append(df)
        else:
            print("No monthly production data for plant " + str(plant_id))
            missing_systems.append(plant_id)
    master_production_df = pd.concat(master_df_list)
    # Get production on a monthly basis for each gen type
    monthly_summed_production = master_production_df.groupby(
        ['sensor_name', 'measured_on_utc'])['value'].sum().reset_index(drop=False)
    monthly_summed_production['sensor_name'] = monthly_summed_production['sensor_name'
                                                                         ].str.replace(" Generation", "")
    monthly_summed_production = monthly_summed_production.pivot(index='measured_on_utc',
                                                                columns='sensor_name', 
                                                                values='value')
    monthly_summed_production['total_generation_mwh'] = monthly_summed_production.sum(axis=1)
    monthly_summed_production.to_csv("monthly_summed_generation_wecc_actual.csv")
    
