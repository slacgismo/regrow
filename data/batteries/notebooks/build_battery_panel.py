import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ============================================================
    # build_battery_panel.py  —  one-stop notebook does the whole thing in one run, for all five years 2018-2022:
    #   EIA-860  -> battery inventory (location + capacity), per year
    #   EIA-923  -> monthly charge / discharge, per year
    #   join     -> one long panel: (plant, month) with capacity + operations
    #
        # Folder layout (paths are resolved from this notebook's location):
        #   battery/raw/EIA860/eia860{YEAR}/2___Plant_Y{YEAR}.xlsx
        #   battery/raw/EIA860/eia860{YEAR}/3_4_Energy_Storage_Y{YEAR}.xlsx
        #   battery/raw/EIA923/EIA923_Schedules_2_3_4_5_M_12_{YEAR}_Final_Revision.xlsx
        #
        # Output: battery/processed/battery_panel_2018_2022.csv
        #         battery/processed/plant_923_not_in_860.csv
    #
    # marimo note: each variable is defined in exactly one cell; multi-step
    # logic lives inside functions.
    # ============================================================
    return


@app.cell
def _():
    # ===== paths =====
    # Every path is resolved from the notebook's own location, so the notebook
    # runs from a fresh clone without editing, and regardless of the working
    # directory marimo was started in.
    from pathlib import Path
    import marimo as mo
    _nb_dir = mo.notebook_dir()
    if _nb_dir is None:                       # running as a plain script
        _nb_dir = Path(globals().get("__file__", ".")).resolve().parent
    BASE = Path(_nb_dir).resolve().parent     # data/batteries/
    RAW = BASE / "raw"                        # inputs, as downloaded
    PROC = BASE / "processed"                 # every table and figure lands here
    PROC.mkdir(parents=True, exist_ok=True)
    return PROC, RAW


@app.cell
def _():
    # ===== imports and constants =====
    import pandas as pd
    import numpy as np

    YEARS = [2018, 2019, 2020, 2021, 2022]

    MONTHS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    # Capacity columns from EIA-860 that may arrive as text and need coercion.
    CAP_COLS = [
        "Nameplate Capacity (MW)",
        "Nameplate Energy Capacity (MWh)",
        "Maximum Charge Rate (MW)",
        "Maximum Discharge Rate (MW)",
    ]

    # Final column layout: identity -> location -> capacity -> time -> operations
    COLUMN_ORDER = [
        "Plant Code", "Plant Name", "State", "County",
        "Latitude", "Longitude", "NERC Region", "Balancing Authority Code",
        "Nameplate Capacity (MW)", "Nameplate Energy Capacity (MWh)",
        "Maximum Charge Rate (MW)", "Maximum Discharge Rate (MW)",
        "Generator Count",
        "Year", "Month",
        "Charge (MWh)", "Discharge (MWh)", "Net Gen (MWh)",
    ]
    return CAP_COLS, COLUMN_ORDER, MONTHS, YEARS, pd


@app.cell
def _(CAP_COLS, RAW, pd):
    # ===== EIA-860 loader (one year -> plant-level battery inventory) =====
    def load_860_batteries(year):
        """Read EIA-860 for one year, keep batteries, aggregate generators up to
        the plant, and attach plant location. Returns one row per battery plant."""
        d = RAW / f"EIA860/eia860{year}"
        storage = pd.read_excel(d / f"3_4_Energy_Storage_Y{year}.xlsx",
                                    sheet_name="Operable", header=1)
        plant = pd.read_excel(d / f"2___Plant_Y{year}.xlsx", header=1)

        # Prime mover BA = battery (excludes flywheel FW, compressed air CP, etc.)
        bat = storage[storage["Prime Mover"] == "BA"].copy()

        # Capacity cells can be blank/text -> force numeric before summing.
        for c in CAP_COLS:
            bat[c] = pd.to_numeric(bat[c], errors="coerce")

        # One plant may hold several battery generators -> sum to plant level.
        agg = bat.groupby("Plant Code")[CAP_COLS].sum(min_count=1)
        agg["Generator Count"] = bat.groupby("Plant Code").size()
        agg = agg.reset_index()

        # Attach location.
        loc = plant[[
            "Plant Code", "Plant Name", "State", "County",
            "Latitude", "Longitude", "NERC Region", "Balancing Authority Code",
        ]]
        out = agg.merge(loc, on="Plant Code", how="left")
        out["Year"] = year
        return out

    return (load_860_batteries,)


@app.cell
def _(MONTHS, RAW, pd):
    # ===== EIA-923 loader (one year -> monthly battery operations, long) =====
    def load_923_batteries(year):
        """Read the Energy Storage tab of EIA-923, keep batteries, and reshape
        the 12 monthly columns into a long table (one row per plant-month)."""
        path = RAW / f"EIA923/EIA923_Schedules_2_3_4_5_M_12_{year}_Final_Revision.xlsx"
        gen = pd.read_excel(path, sheet_name="Page 1 Energy Storage", header=5)

        # Column names carry embedded newlines ('Reported\nPrime Mover'); flatten.
        gen.columns = [str(c).replace("\n", " ").strip() for c in gen.columns]
        bat = gen[gen["Reported Prime Mover"] == "BA"].copy()

        # Wide -> long. EIA treats electricity as the fuel, so:
        #   Quantity = gross charge, Grossgen = gross discharge,
        #   Netgen   = discharge - charge (usually negative).
        frames = []
        for i, m in enumerate(MONTHS, start=1):
            sub = bat[["Plant Id"]].copy()
            sub["Month"] = i
            sub["Charge (MWh)"] = pd.to_numeric(bat[f"Quantity {m}"], errors="coerce")
            sub["Discharge (MWh)"] = pd.to_numeric(bat[f"Grossgen {m}"], errors="coerce")
            sub["Net Gen (MWh)"] = pd.to_numeric(bat[f"Netgen {m}"], errors="coerce")
            frames.append(sub)

        long = pd.concat(frames, ignore_index=True)
        return bat, long

    return (load_923_batteries,)


@app.cell
def _(load_860_batteries, load_923_batteries):
    # ===== merge one year (860 capacity + 923 operations) =====
    def merge_year(year):
        """Left join, 860 as the anchor (only 860 carries coordinates).
        Prints the match diagnostics and returns (merged, orphan_plant_ids)."""
        cap = load_860_batteries(year)
        bat923, ops = load_923_batteries(year)

        merged = cap.merge(
            ops, left_on="Plant Code", right_on="Plant Id", how="left"
        )

        in_860 = set(cap["Plant Code"])
        in_923 = set(bat923["Plant Id"])
        print(f"--- {year} ---")
        print(f"  860 battery plants        : {len(in_860)}")
        print(f"  923 battery plants        : {len(in_923)}")
        print(f"  matched                   : {len(in_860 & in_923)}")
        print(f"  only in 860 (no ops data) : {len(in_860 - in_923)}")
        print(f"  only in 923 (no coords)   : {len(in_923 - in_860)}")

        return merged, sorted(in_923 - in_860) 

    return (merge_year,)


@app.cell
def _(COLUMN_ORDER, YEARS, merge_year, pd):
    # ===== build the full five-year panel =====
    def build_panel(years):
        """Loop the years, stack, clean types, drop the duplicate join key,
        and apply the final column layout. Returns (panel, orphan_log)."""
        frames = []
        orphans = {}
        for y in years:
            m, o = merge_year(y)
            frames.append(m)
            orphans[y] = o

        raw = pd.concat(frames, ignore_index=True)

        # Coordinates can arrive as text (blank source rows) -> numeric.
        raw["Latitude"] = pd.to_numeric(raw["Latitude"], errors="coerce")
        raw["Longitude"] = pd.to_numeric(raw["Longitude"], errors="coerce")

        # 'Plant Id' (923 side of the join) duplicates 'Plant Code' -> drop.
        tidy = (
            raw.drop(columns=["Plant Id"])[COLUMN_ORDER]
            .sort_values(["Plant Code", "Year", "Month"])
            .reset_index(drop=True)
        )
        return tidy, orphans


    panel, orphan_log = build_panel(YEARS)

    print("\nPanel shape:", panel.shape)
    print("Rows missing coordinates:", panel["Latitude"].isna().sum())
    panel.head(12)
    return orphan_log, panel


@app.cell
def _(panel):
    # ===== sanity check - annual totals =====
    # Capacity is constant within a plant-year -> take first per plant before sum.
    cap_by_year = (
        panel.groupby(["Year", "Plant Code"])["Nameplate Capacity (MW)"]
        .first()
        .groupby("Year")
        .sum()
    )

    summary = panel.groupby("Year").agg(
        plants=("Plant Code", "nunique"),
        charge_MWh=("Charge (MWh)", "sum"),
        discharge_MWh=("Discharge (MWh)", "sum"),
    ).round(0)

    summary.insert(1, "MW", cap_by_year.round(1))
    summary["round_trip_%"] = (summary["discharge_MWh"] / summary["charge_MWh"] * 100).round(1)
    summary
    return


@app.cell
def _(PROC, orphan_log, panel, pd):
    # ===== save =====
    panel.to_csv(PROC / "battery_panel_2018_2022.csv", index=False)
    print("Saved: battery_panel_2018_2022.csv")
    print("Rows:", len(panel), "| Plants:", panel["Plant Code"].nunique())

    orphan_rows = [{"Year": y, "Plant Id": p} for y, ps in orphan_log.items() for p in ps]
    pd.DataFrame(orphan_rows).to_csv(PROC / "plant_923_not_in_860.csv", index=False)
    print("Saved: plant_923_not_in_860.csv |", len(orphan_rows), "plant-years")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
