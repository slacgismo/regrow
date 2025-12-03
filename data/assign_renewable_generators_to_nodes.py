import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    This notebook contained the process for assigning nodes to USPVDB and USWTDB generators.

    1. Load WECC node summary with solar and wind assignment
    2. Load USPVDB and USWTDB system data
    3. Assign each generator to the closest node in the available set

    ### Notebook output

    - `pv_generators_assigned.csv`: contains a table of USPVDB systems that are in the WECC service area, with their node geohash added
    - `wt_generators_assigned.txt`: contains a table of USWTDB systems that are in the WECC service area, with their node geohash added
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import utils

    def add_node_hashes(gen_df, nodes_df):
        df = gen_df.copy()
        latlon_list = [
            (nodes_df.iloc[_ix]['Lat'], nodes_df.iloc[_ix]['Long']) 
            for _ix in range(len(nodes_df))
        ]
        hash_list = []
        for _ix in mo.status.progress_bar(range(len(df))):
            _best_ix, _latlon, _dist = utils.nearest2(
                (df.iloc[_ix]['latitude_gen'], df.iloc[_ix]['longitude_gen']), 
                latlon_list
            )
            hash_list.append(nodes_df.iloc[_best_ix].name)
        df['geohash'] = hash_list
        return df
    return add_node_hashes, mo, pd, utils


@app.cell
def _(pd, utils):
    reduced_network = utils.load_reduced_network()
    wecc_wt_systems = pd.read_csv('wecc_wt_systems.csv', index_col=0)
    wecc_pv_systems = pd.read_csv('wecc_pv_systems.csv', index_col=0)
    return reduced_network, wecc_pv_systems, wecc_wt_systems


@app.cell
def _(add_node_hashes, reduced_network, wecc_pv_systems):
    pv_generators_assigned = add_node_hashes(wecc_pv_systems, reduced_network[reduced_network['pv_gen']])
    return (pv_generators_assigned,)


@app.cell
def _(add_node_hashes, reduced_network, wecc_wt_systems):
    wt_generators_assigned = add_node_hashes(wecc_wt_systems, reduced_network[reduced_network['wt_gen']])
    return (wt_generators_assigned,)


@app.cell
def _(pv_generators_assigned, wt_generators_assigned):
    wt_generators_assigned.to_csv('wt_generators_assigned.csv')
    pv_generators_assigned.to_csv('pv_generators_assigned.csv')
    return


@app.cell
def _(pv_generators_assigned):
    len(set(pv_generators_assigned['geohash']))
    return


if __name__ == "__main__":
    app.run()
