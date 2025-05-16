"""US region data for interconnections

* `systems`: specifies which regions are in interconnected systems

* `regions`: specifies which states are in regions

* `exclude`: specifies which counties are not included in regions

* `include`: specifies which counties are included in regions

* `counties`: provides counties.csv with regions and systems

* `interconnection`: maps regions to interconnections

"""
import pandas as pd
import states

systems = {
    "Eastern" : ["MRO","NPCC","RF","SERC"],
    "Western" : ["WECC"],
    "Texas": ["ERCOT"],
    "Alaska": ["Alaska"],
    "Hawaii": ["Hawaii"],
    "Puerto Rico" : ["Puerto Rico"],
}
regions = {
    "US" : [x[1] for x in states.state_codes],
    "Alaska": ["AK"],
    "ERCOT" : ["TX"],
    "Hawaii": ["HI"],
    "Puerto Rico": ["PR"],
    "MRO" : ["AR","IA","KS","LA","MN","MO","ND","NE","NM","OK","SD","TX","WI","CO","MT",],
    "NPCC" : ["CT","MA","ME","NH","NY","RI","VT"],
    "RF" : ["DC","DE","IL","WI","IN","KY","MD","MI","NJ","OH","PA","VA","WI","WV"],
    "SERC" : ["AR","AL","FL","IL","GA","KY","LA","MS","MO","NC","OK","SC","TN","VA"],
    "WECC" : ["AZ","CA","CO","ID","MT","NM","NV","OR","SD","TX","UT","WA","WY"],
}

exclude = {
    "ERCOT" : {
        "TX": ["48141"],
    },
    "WECC" : {
        "CO" : ["08095","08115"],
        "MT" : ["30021","30025","30109"],
        "NM" : ["35009","30019","35025","35037","35041","30083","30085"],
    },
}

include = {
    "WECC" : {
        "SD" : ["46019","46033","46047","46081","46103"],
        "TX" : ["48141"],
    },
    "MRO" : {
        "CO" : ["08095","08115"],
        "MT" : ["30021","30025","30109"],
        "NM" : ["35009","30019","35025","35037","35041","30083","30085"],
    }
}

counties = pd.read_csv("counties.csv",index_col=["usps"],dtype=str)

# for county in counties.iterrows():

#     region = [x for x,y in regions.items() if counties.usps.values in y]
#     print(county,"-->",region)

counties["region"] = ""
for region,states in [(x,y) for x,y in regions.items() if x != "US"]:
    for state in states:
        if region in include and state in include[region]:
            counties.loc[counties.fips.isin(include[region][state]),"region"] = region
        elif region in exclude and state in exclude[region]:
            counties.loc[(counties.index==state)&(~counties.fips.isin(exclude[region][state])),"region"] = region
        else:
            counties.loc[state,"region"] = region

interconnection = {}
for x,y in systems.items():
    for z in y:
        interconnection[z] = x

counties["system"] = [interconnection[x] for x in counties["region"]]

counties.reset_index(inplace=True)
