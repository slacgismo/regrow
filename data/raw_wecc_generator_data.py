import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    This notebook demonstrates the process for assigning generator types to WECC 240 model nodes.

    #### Inputs:

    - `wecc240/wecc240raw_generators.csv`
    - `wecc240_gis.csv`, via `utils.load_full_network()`

    #### Outputs:

    - `wecc240/wecc_generators_with_bus_geohash.csv`
    - `wecc240/pv_nodes.csv`
    - `wecc240/wt_nodes.csv`

    ### Generator type codes

    - Biomass: NB, B
    - Gas: G, DG, EG, TG, RG, SG, WG, NG, MG, CG
    - Geothermal: CE, NE
    - Hydro: H, NH
    - Nuclear: N, NN
    - PV: DP, S
    - Steam: C, E,
    - Wind: W, NW, SW
    - BESS: BESS, BA
    - PSH: PSH
    - R is renewable (not sure which type).
    - DC is HVDC line but modeled as a generator.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import utils
    return mo, pd, utils


@app.cell
def _(pd):
    wecc_generators = pd.read_csv('wecc240/wecc240raw_generators.csv')
    return (wecc_generators,)


@app.cell
def _(utils):
    network = utils.load_full_network()
    return (network,)


@app.cell
def _(network, pd, wecc_generators):
    joined = pd.merge(wecc_generators, network, how='inner', left_on='   I', right_on='Bus  Number')
    return (joined,)


@app.cell
def _(mo, save_file, save_pv, save_wt):
    mo.hstack([save_file, save_pv, save_wt])
    return


@app.cell
def _(mo):
    save_file = mo.ui.run_button(label='Save joined file')
    save_pv = mo.ui.run_button(label='Save pv file')
    save_wt = mo.ui.run_button(label='Save wt file')
    return save_file, save_pv, save_wt


@app.cell
def _(joined, mo, save_file):
    mo.stop(not save_file.value)
    joined.to_csv('wecc240/wecc_generators_with_bus_geohash.csv')
    print('file saved!')
    return


@app.cell
def _(joined):
    joined
    return


@app.cell
def _(joined):
    joined[joined['ID'].isin(['R'])]
    return


@app.cell
def _(joined):
    _grouped = joined[joined['ID'].isin(['DP', 'S'])]\
        .groupby('geohash')
    pv_nodes = _grouped.first()
    pv_nodes
    return (pv_nodes,)


@app.cell
def _(mo, pv_nodes, save_pv):
    mo.stop(not save_pv.value)
    pv_nodes.to_csv('wecc240/pv_nodes.csv')
    print('file saved!')
    return


@app.cell
def _(joined):
    _grouped = joined[joined['ID'].isin(['W', 'NW', 'SW'])]\
        .groupby('geohash')
    wt_nodes = _grouped.first()
    wt_nodes
    return (wt_nodes,)


@app.cell
def _(mo, save_wt, wt_nodes):
    mo.stop(not save_wt.value)
    wt_nodes.to_csv('wecc240/wt_nodes.csv')
    print('file saved!')
    return


@app.cell
def _(joined, pd):
    # Data frame of node geohashes that have either solar or wind generation
    # The "I" column is the node ID from the original WECC model
    # the "ID" column is the type of generator
    _grouped = joined[joined['ID'].isin(['W', 'NW', 'SW', 'DP', 'S', 'R'])]\
        .groupby('geohash')
    pd.merge(_grouped['   I'].apply(set), _grouped['ID'].apply(list), left_index=True, right_index=True)
    return


if __name__ == "__main__":
    app.run()
