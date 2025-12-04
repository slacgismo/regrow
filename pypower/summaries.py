"""This script produces various summaries of the model needed for the
study. All output is written the 'summaries' folder.

This script should be rerun anytime one of the model inputs or libraries
is changed. This can be done by running `make summaries`.
"""

import os
import warnings
from pypower_sim.ppmodel import PPModel
from wecc240 import wecc240
import eia860m
import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None
pd.options.display.max_rows = None

os.makedirs("summaries",exist_ok=True)

def gis_2011(options) -> pd.DataFrame:
    """Get the GIS data for the 2011 model"""
    model = PPModel(case=wecc240)
    return model.get_data("gis").set_index("GEOHASH")

def gis_2020(options) -> pd.DataFrame:
    """Get the GIS data for 2020 model"""
    model = PPModel(case=wecc240(options=options))
    return model.get_data("gis").set_index("GEOHASH")

def node_gencount(options) -> pd.DataFrame:
    """Get gis GEN and LOAD counts by GEOHASH"""
    model = PPModel(case=wecc240(options=options))
    return model.get_data("gis").set_index("GEOHASH").sort_index()["GEN"].dropna().rename({"GEN":"GENCOUNT"}).groupby("GEOHASH").sum().astype(int)

def bus_catalog(options) -> pd.DataFrame:
    """Generate the full bus catalog"""
    model = PPModel(case=wecc240(options=options))
    data = pd.merge(model.get_data("bus"),model.get_data("gis"),left_on="BUS_I",right_on="BUS_I")
    data = pd.merge(data,model.get_data("gen"),left_on="BUS_I",right_on="GEN_BUS",how="outer").drop("GEN_BUS",axis=1)
    data["BUS_I"] = data.BUS_I.astype(int)
    data["BUS_TYPE"] = [["NONE","PQ","PV","REF"][round(x)] for x in data.BUS_TYPE.astype(int)]
    data.set_index(["GEOHASH","BUS_I"],inplace=True)
    data = data[["NAME","BUS_TYPE","BASE_KV","PD","PMAX"]]
    data.columns = ["NAME","BUS_TYPE","VOLTAGE","LOAD","GENERATION"]
    data["GENOK"] = 0
    data.loc[data.BUS_TYPE=="PV","GENOK"] = 1
    return data

def bus_nogen(options) -> pd.DataFrame:
    """Generate the list of busses that cannot accept new generation"""
    data = bus_catalog(options)
    n_genbus = pd.DataFrame(data[["GENOK","LOAD"]].groupby("GEOHASH").sum()) # count of how many PV busses are there
    return data.reset_index().set_index("GEOHASH").loc[(n_genbus.GENOK==0)&(n_genbus.LOAD==0)]

def network_bus_graph(options):
    """Get network bus graph"""
    model = PPModel(case=wecc240(options=options))
    return pd.DataFrame(model.get_graph("BUS")).rename({0:"FROM",1:"TO"},axis=1).set_index("FROM")

def network_node_graph(options):
    """Get network bus graph"""
    model = PPModel(case=wecc240(options=options))
    return pd.DataFrame(model.get_graph("GEOHASH")).rename({0:"FROM",1:"TO"},axis=1).set_index("FROM")

def network_zone_graph(options):
    """Get network bus graph"""
    model = PPModel(case=wecc240(options=options))
    return pd.DataFrame(model.get_graph("ZONE")).rename({0:"FROM",1:"TO"},axis=1).set_index("FROM")

def network_area_graph(options):
    """Get network bus graph"""
    model = PPModel(case=wecc240(options=options))
    return pd.DataFrame(model.get_graph("AREA")).rename({0:"FROM",1:"TO"},axis=1).set_index("FROM")

def bus_generator_histogram(options):
    """Generate histogram of generation according to bus voltage"""
    model = PPModel(case=wecc240(options=options))
    data = pd.merge(model.get_data("bus"),model.get_data("gen"),left_on="BUS_I",right_on="GEN_BUS")
    grouper = data.groupby(["BUS_I","BASE_KV"])["PMAX"]
    result = pd.concat([grouper.sum().round(1),grouper.count()],axis=1)
    result.columns = ["PMAX","COUNT"]
    index_names = result.index.names
    result.reset_index(inplace=True)
    result["BUS_I"] = result.BUS_I.astype(int)
    return result.set_index(index_names).groupby("BASE_KV").sum()

def bus_voltage_class(options):
    """Report bus voltage classes, i.e., HV, MV, or LV"""
    voltage_ranges = {"LV":[0,50.0],"MV":[50,250],"HV":[250,1000]}
    model = PPModel(case=wecc240(options=options))
    data = model.get_data("bus").copy()
    data["BUS_I"] = data.BUS_I.astype(int)
    data["VCLASS"] = "NONE" # default class is NONE
    def get_class(v):
        for vc,vr in voltage_ranges.items():
            if vr[0] < v <= vr[1]:
                return vc
        return NONE
    data["VCLASS"] = [get_class(x) for x in data.BASE_KV]
    return data.set_index("BUS_I")[["VCLASS"]]

def eia860m_node_assignment(options):
    """Generation EIA Form 860 assignment summary and KML"""
    eia860 = eia860m.EIA860(reload=False)
    casedata = wecc240()

    pd.options.display.max_columns = None
    pd.options.display.width = None

    # test loading gen data into WECC 240 case
    gen = eia860.to_gen(
        case=casedata,
        converters={"fuel":eia860m.FUELS,"gen":eia860m.GENS},
        index_csv="summaries/eia860m_nodes.csv",
        )

    # test load gencost data into WECC240 case
    gencost = eia860.to_gencost(
        case=casedata,
        )

    casedata["gen"] = gen.values
    casedata["gencost"] = gencost.values

    from pypower.runpf import runpf
    from pypower.runopf import runopf
    from pypower.ppoption import ppoption

    result = {}

    pf,status = runpf(casedata,ppoption(VERBOSE=0,OUT_ALL=0))
    result["Powerflow time"] = f"{pf['et']*1000:.1f} ms" if status else 'FAILED'
    if status == 0:
        warnings.warn("EIA860m powerflow solution failed")

    opf = runopf(casedata,ppoption(VERBOSE=0,OUT_ALL=0))
    result["AC OPF stime"] = f"{opf['et']*1000:.1f} ms" if opf['success'] else 'FAILED'
    if opf['success'] == 0:
        warnings.warn("EIA860m AC OPF solution failed")
    opfpf,status = runpf(opf,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        warnings.warn("EIA860m AC OPF powerflow solution failed")
    result["AC OPF powerflow time"] = f"{opfpf['et']*1000:.1f} ms" if status else 'FAILED'


    gen.index.names = [
        "States",
        "Counties",
        "Nodes",
        "Busses",
        "Fuel types",
        "Generator types",
        ]
    for level in set(gen.index.names):
        data = gen.PMAX.groupby(level).sum().sort_values(ascending=False).to_frame()
        result[level] = len(data)
    result["Total generators"] = len(gen)
    result["Total capacity (GW)"] = round(float(gen.PMAX.sum()/1000),1)
    result["Operating cost ($M)"] = f"{opf["f"]*casedata["baseMVA"]/1000:.1f}"

    result = pd.DataFrame(result.values(),result.keys(),columns=["Result"])
    result.index.name = "EIA860m Summary"

    eia860.to_kml("summaries/eia860m_nodes.kml")

    return result


if __name__ == "__main__":

    model_options = ["SCHEDULING"]
    for summary in [
            "gis_2011","gis_2020",
            "node_gencount",
            "bus_catalog","bus_nogen","bus_generator_histogram","bus_voltage_class",
            "network_bus_graph", "network_node_graph","network_zone_graph","network_area_graph",
            "eia860m_node_assignment"
            ]:
        globals()[summary](options=model_options).to_csv(f"summaries/{summary}.csv")

