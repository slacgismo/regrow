import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ============================================================
    # nearest_node.py  —  cell-by-cell content for marimo
    #
    # Goal: keep WECC battery plants only (the WEC-240 model is a WECC
    #       model), label each with the nearest node by great-circle
    #       distance, and attach that geohash label onto the monthly panel.
    #       The match distance is kept as a data-quality reference.
    #
    # Repo layout (this notebook lives in data/batteries/notebooks/):
    #   data/batteries/
    #     raw/utils.py    - haversine_distance(), geohash(), nearest()
    #     raw/nodes.csv   - 126 unique nodes, with geocode + Lat/Long
    #     processed/      - panel in; labeled panel and node tables out
    #     notebooks/      - this file
    #
    # Paths resolve relative to this file, so it runs from a fresh clone
    # regardless of the working directory.
    #
    # Produces the two deliverables:
    #   processed/node_capacity_by_year.csv     - per (geohash, year)
    #   processed/node_generation_by_month.csv  - per (geohash, year, month)
    #
    # marimo note: each variable is defined in exactly one cell.
    # ============================================================
    return


@app.cell
def _():
    # ===== imports and paths =====
    import os, sys, types
    from pathlib import Path

    # utils.py reads HOME at import time and imports psm3 at the top.
    # Windows has neither, so provide a HOME fallback and stub psm3 out.
    # Environment shim only - changes no calculation.
    os.environ.setdefault("HOME", os.environ.get("USERPROFILE", os.path.expanduser("~")))
    sys.modules.setdefault("psm3", types.ModuleType("psm3"))

    try:
        HERE = Path(__file__).resolve().parent      # data/batteries/notebooks
    except NameError:
        HERE = Path.cwd()
    BATT = HERE.parent                              # data/batteries
    RAW = BATT / "raw"
    PROC = BATT / "processed"
    PROC.mkdir(exist_ok=True)

    # utils.py lives in raw/ alongside the node files it works with.
    sys.path.insert(0, str(RAW))

    import pandas as pd
    import numpy as np
    from utils import haversine_distance
    from utils import geohash, nearest

    print("raw       :", RAW)
    print("processed :", PROC)
    return PROC, RAW, geohash, nearest, pd


@app.cell
def _(PROC, RAW, pd):
    # ===== load inputs and keep WECC plants only =====
    # The WEC-240 model covers the Western Interconnection (WECC), so we
    # restrict to plants whose NERC Region is WECC. This is the electrically
    # correct footprint - unlike a pure distance cut, it excludes Texas (ERCOT)
    # and Midwest (MRO) plants that happen to sit near the western edge.

    panel_all = pd.read_csv(PROC / "battery_panel_2018_2022.csv")
    nodes = pd.read_csv(RAW / "nodes.csv")

    panel = panel_all[panel_all["NERC Region"] == "WECC"].copy().reset_index(drop=True)

    print("Monthly rows: all", len(panel_all), "-> WECC", len(panel))
    print("Node locations:", len(nodes))
    return nodes, panel


@app.cell
def _(panel):
    # ===== one row per battery plant (coordinates are per-plant) =====
    plants = (
        panel.drop_duplicates(subset="Plant Code")[
            ["Plant Code", "Plant Name", "State", "Latitude", "Longitude"]
        ]
        .reset_index(drop=True)
    )
    print("WECC battery plants:", len(plants))
    return (plants,)


@app.cell
def _(geohash, nearest, nodes, plants):
    # ===== nearest-node search using the repo's own functions from utils.py =====
    # Encode each plant's lat/long as a geohash, then find the closest node
    # geohash. Keeps the whole match on the repo's geohash convention rather
    # than introducing a second distance implementation.
    node_hashes = nodes["geocode"].tolist()

    def match_plant(row):
        plant_hash = geohash(row["Latitude"], row["Longitude"])
        nearest_hash, dist_m = nearest(plant_hash, node_hashes, withdist=True)
        return nearest_hash, round(dist_m / 1000.0, 2)   # (node geohash, distance in KM)

    matched = plants.copy()
    matched[["geohash", "match_dist_km"]] = matched.apply(
        match_plant, axis=1, result_type="expand"
    )
    return (matched,)


@app.cell
def _(matched):
    # ===== match-distance distribution (data-quality reference) =====
    # Distance is kept purely as a quality flag now, not a filter. Most WECC
    # plants sit within a few tens of km of a node. A handful in AZ/NM/CO are
    # 100-450 km out because the WEC-240 model has sparse nodes in the interior
    # West - that is model resolution, not a coordinate error.
    print(matched["match_dist_km"].describe().round(1))
    print("\nPlants matched >100 km from nearest node:", (matched["match_dist_km"] > 100).sum())
    print(matched[matched["match_dist_km"] > 100]
          [["Plant Name", "State", "match_dist_km"]]
          .sort_values("match_dist_km", ascending=False)
          .to_string(index=False))
    return


@app.cell
def _(matched, panel):
    # ===== attach geohash label back onto the monthly panel =====
    labeled = panel.merge(
        matched[["Plant Code", "geohash", "match_dist_km"]],
        on="Plant Code",
        how="left",
    )
    print("Labeled panel shape:", labeled.shape)
    print("Plants:", labeled["Plant Code"].nunique())
    print("Rows with a geohash:", labeled["geohash"].notna().sum())
    labeled.head(6)
    return (labeled,)


@app.cell
def _(PROC, labeled, matched):
    # ===== save the labeled panel and the plant -> node map =====
    labeled.to_csv(PROC / "battery_panel_labeled.csv", index=False)
    matched.to_csv(PROC / "plant_to_node.csv", index=False)
    print("Saved:", PROC / "battery_panel_labeled.csv", " (WECC monthly panel + geohash label)")
    print("Saved:", PROC / "plant_to_node.csv", " (WECC plants: node + distance)")
    return


@app.cell
def _(labeled):
    # ===== CAPACITY table, aggregated by node and year =====
    # Capacity is a per-plant-year attribute (does not change month to month),
    # so take the first value per (node, year, plant) before summing across the
    # plants that share a node - otherwise the 12 monthly rows would multiply it.
    _cap_cols = [
        "Nameplate Capacity (MW)",
        "Nameplate Energy Capacity (MWh)",
        "Maximum Charge Rate (MW)",
        "Maximum Discharge Rate (MW)",
    ]

    _plant_year_cap = (
        labeled.groupby(["geohash", "Year", "Plant Code"])[_cap_cols].first()
    )

    capacity = _plant_year_cap.groupby(["geohash", "Year"]).sum().round(1).reset_index()
    capacity["Plant Count"] = (
        labeled.groupby(["geohash", "Year"])["Plant Code"].nunique().values
    )

    print("CAPACITY table:", capacity.shape, "| nodes:", capacity["geohash"].nunique())
    print("Total MW by year:")
    print(capacity.groupby("Year")["Nameplate Capacity (MW)"].sum().round(1).to_string())
    capacity.head(10)
    return (capacity,)


@app.cell
def _(labeled):
    # ===== GENERATION table, aggregated by node, year, month (long) =====
    # One row per (node, year, month) with the three operational metrics.
    # min_count=1 keeps a NaN (rather than 0) when every plant at that node-month
    # was missing 923 data, so "no report" stays distinguishable from "zero".
    generation = (
        labeled.groupby(["geohash", "Year", "Month"])[
            ["Charge (MWh)", "Discharge (MWh)", "Net Gen (MWh)"]
        ]
        .sum(min_count=1)
        .round(1)
        .reset_index()
    )
    print("GENERATION table:", generation.shape, "| nodes:", generation["geohash"].nunique())
    generation.head(12)
    return (generation,)


@app.cell
def _(PROC, capacity, generation):
    # ===== save both deliverables =====
    capacity.to_csv(PROC / "node_capacity_by_year.csv", index=False)
    generation.to_csv(PROC / "node_generation_by_month.csv", index=False)
    print("Saved:", PROC / "node_capacity_by_year.csv", "(", len(capacity), "rows )")
    print("Saved:", PROC / "node_generation_by_month.csv", "(", len(generation), "rows )")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
