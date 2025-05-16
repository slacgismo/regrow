"""County mapping to nodes

* `data`: provides counties.csv with regions and systems

* `systems`: specifies which regions are in interconnected systems

* `regions`: specifies which states are in regions

* `exclude`: specifies which counties are not included in regions

* `include`: specifies which counties are included in regions

* `interconnection`: maps regions to interconnections

"""

import pandas as pd
import utils
import states
import load_models as lm

def find_node(county:str,nodes:list[str]) -> str:
    """Find node nearest to county centroid"""
    if isinstance(county,lm.County):
        return utils.nearest(county.geocode,nodes)
    elif isinstance(county,str):
        return utils.nearest(county,nodes)
    raise ValueError(f"{county=} is not a County load_model object or geocode")

def find_nodes(county,geocodes,nodes):
    """Find nodes that counties are nearest"""
    return [y for y in nodes if county in [x for x in find_counties(y,geocodes,nodes) if county in x]]

def find_counties(
    node:str, # node to search for
    geocodes:list[str], # counties geocodes
    nodes:list[str], # node list
    ) -> list[str]:
    """Get counties that map to node"""
    result = []
    for geocode in geocodes:
        if utils.nearest(geocode,nodes) == node:
            result.append(lm.County(geocode))
    return result

def county_mapping(counties:list[str],nodes:list[str]) -> dict[str:list[str]]:
    """Find all the counties that map to each node in the nodes list

    Arguments:

    * `counties`: list of counties of map to nodes

    * `nodes`: list of nodes to which counties must be mapped

    Returns:

    * `dict`: nodes to which counties are mapped
    """
    return {y:[x for x in find_counties(y,list(counties),nodes)] for y in nodes}

    # counties = {}
    # for node in nodes:
    #     counties[node] = [x for x in find_counties(node,list(wecc),nodes)]

    # return counties

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

interconnection = {}
for x,y in systems.items():
    for z in y:
        interconnection[z] = x

data = pd.read_csv("counties.csv",index_col=["usps"],dtype=str)
data["region"] = ""
for region,states in [(x,y) for x,y in regions.items() if x != "US"]:
    for state in states:
        if region in include and state in include[region]:
            data.loc[data.fips.isin(include[region][state]),"region"] = region
        elif region in exclude and state in exclude[region]:
            data.loc[(data.index==state)&(~data.fips.isin(exclude[region][state])),"region"] = region
        else:
            data.loc[state,"region"] = region

data["system"] = [interconnection[x] for x in data["region"]]
data.reset_index(inplace=True)

if __name__ == "__main__":

    # nodes = list(pd.read_csv("geodata/temperature.csv",index_col="timestamp").columns)
    # wecc = {x:y for x,y in lm.County._counties.iterrows() if y.usps in lm.County._regions["WECC"]}
    
    test = data.set_index(["region","usps"]).sort_index()
    assert list(test.loc[("WECC","TX")]["fips"].values) == ['48141'], "TX WECC result incorrect"
    assert list(test.loc[("WECC","SD")]["fips"].values) == ['46019', '46033', '46047', '46081', '46103'], "SD WECC result incorrect"

    # print(nodes)

    # print("County --> Code")
    # print("---------------")
    # for geocode,county in wecc.items():
    #     node = find_node(geocode,nodes)
    #     print(county.county,county.usps,"-->",node)

    # print("Node --> Counties")
    # print("-----------------")
    # counties = {}
    # for node in nodes:
    #     # print(node,"-->")
    #     counties[node] = [x for x in find_counties(node,list(wecc),nodes)]
    #     # for name,data in counties[node].items():
    #         # print(f"    {data}",flush=True)

    # print(counties)

    # print("County --> Nodes")
    # print("----------------")
    # for county in list(wecc):
    #     print(county,"-->",find_nodes(county,list(wecc),nodes),flush=True)

    # print(county_mapping(wecc,nodes))