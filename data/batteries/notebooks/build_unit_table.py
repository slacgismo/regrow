import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ============================================================
    # build_unit_table.py  —  generator-level battery attributes, 2018-2022, in OP or SB(standby/backup)
    #
    # Why a separate notebook: build_battery_panel.py aggregates to the plant
    # immediately (groupby -> sum), which is correct for capacity but destroys
    # the per-generator attributes we now need. Storage technology, enclosure
    # type, reactive power rating and the eleven application flags are all
    # generator-level facts, and several plants hold more than one battery
    # generator. This notebook keeps that grain.
    #
    # Inputs (all in gis_match/):
    #   EIA860/eia860{YEAR}/3_4_Energy_Storage_Y{YEAR}.xlsx
    #   plant_to_node.csv         - WECC plants with their node (nearest2)
    #   node_capacity_by_year.csv - existing deliverable, used only to validate
    #     the pre-filter aggregation; it predates the dispatchable-status
    #     filter applied here and is therefore wider than the output
    #
    # Output: battery_units_2018_2022.csv
    #
    # ============================================================
    return


@app.cell
def _():
    # ===== imports and constants =====
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

    import pandas as pd
    import numpy as np

    YEARS = [2018, 2019, 2020, 2021, 2022]

    # Identity and location fields carried through unchanged.
    ID_COLS = [
        "Utility ID", "Utility Name", "Plant Code", "Plant Name",
        "State", "County", "Generator ID", "Status",
        "Technology", "Prime Mover", "Operating Month", "Operating Year",
    ]

    # Numeric fields. These arrive as text on blank source rows, so every one
    # of them goes through to_numeric(errors="coerce").
    NUM_COLS = [
        "Nameplate Capacity (MW)",
        "Nameplate Energy Capacity (MWh)",
        "Maximum Charge Rate (MW)",
        "Maximum Discharge Rate (MW)",
        "Nameplate Reactive Power Rating",
    ]

    # Storage Technology 1-4 hold codes from EIA-860 Instructions Table 5b.
    TECH_COLS = [f"Storage Technology {i}" for i in (1, 2, 3, 4)]

    TECH_NAMES = {
        "LIB": "Lithium-ion battery",
        "NAB": "Sodium based battery",
        "PBB": "Lead-acid battery",
        "NIB": "Nickel based battery",
        "FLB": "Flow battery",
        "ECC": "Electro-chemical capacitor",
        "MAB": "Metal air battery",
        "OTH": "Other",
    }

    ENCLOSURE_NAMES = {
        "BL": "Building",
        "CS": "Containerized - stationary",
        "CT": "Containerized - transportable",
        "OT": "Other",
    }

    # The eleven Y/N application flags (Instructions line 40). Reported
    # voluntarily, so blanks are common and must stay distinct from "N".
    APP_COLS = [
        "Arbitrage",
        "Frequency Regulation",
        "Load Following",
        "Ramping / Spinning Reserve",
        "Co-Located Renewable Firming",
        "Transmission and Distribution Deferral",
        "System Peak Shaving",
        "Load Management",
        "Voltage or Reactive Power Support",
        "Backup Power",
        "Excess Wind and Solar Generation",
    ]

    OTHER_COLS = ["Storage Enclosure Type"]

    WANTED = ID_COLS + NUM_COLS + TECH_COLS + OTHER_COLS + APP_COLS
    return (
        APP_COLS,
        ENCLOSURE_NAMES,
        NUM_COLS,
        PROC,
        RAW,
        TECH_COLS,
        TECH_NAMES,
        WANTED,
        YEARS,
        pd,
    )


@app.cell
def _():
    # ===== column-name resolution =====
    # EIA renames columns slightly between vintages: stray whitespace, embedded
    # newlines, and unit suffixes such as "(MVAR)" appear and disappear. Match
    # on a normalised key and on a prefix so one loader works for all years.
    import re

    def norm_key(name):
        """Lowercase, strip punctuation and collapse whitespace."""
        s = str(name).replace("\n", " ")
        s = re.sub(r"[^0-9a-zA-Z ]+", " ", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    def resolve(wanted, available):
        """Map each wanted column to the actual column present in the sheet.

        Returns (mapping, missing). Exact normalised match is tried first, then
        a prefix match so 'Nameplate Reactive Power Rating' also picks up
        'Nameplate Reactive Power Rating (MVAR)'.
        """
        lookup = {}
        for col in available:
            lookup.setdefault(norm_key(col), col)

        mapping, missing = {}, []
        for w in wanted:
            k = norm_key(w)
            if k in lookup:
                mapping[w] = lookup[k]
                continue
            hits = [orig for key, orig in lookup.items() if key.startswith(k)]
            if len(hits) == 1:
                mapping[w] = hits[0]
            else:
                missing.append(w)
        return mapping, missing

    return (resolve,)


@app.cell
def _(NUM_COLS, RAW, WANTED, pd, resolve):
    # ===== EIA-860 loader (one year -> generator-level battery rows) =====
    def load_860_units(year):
        """Read the Operable sheet, keep battery generators, and return one row
        per generator with the attribute columns resolved and typed.

        Prime Mover BA is the battery filter. The other storage prime movers on
        this sheet are FW flywheel, CE compressed air, CP concentrated solar
        power and PS pumped storage, none of which are in scope.
        """
        path = RAW / f"EIA860/eia860{year}/3_4_Energy_Storage_Y{year}.xlsx"
        raw = pd.read_excel(path, sheet_name="Operable", header=1)
        raw.columns = [str(c).replace("\n", " ").strip() for c in raw.columns]

        mapping, missing = resolve(WANTED, raw.columns)

        bat = raw[raw["Prime Mover"] == "BA"].copy()
        out = pd.DataFrame(index=bat.index)
        for want, actual in mapping.items():
            out[want] = bat[actual]
        for want in missing:
            out[want] = pd.NA

        for c in NUM_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out["Year"] = year
        return out.reset_index(drop=True), missing

    return (load_860_units,)


@app.cell
def _(WANTED, YEARS, load_860_units, pd):
    # ===== build the five-year unit table =====
    def build_units(years):
        """Stack every year and report what was found, so a silently missing
        column in one vintage cannot pass unnoticed."""
        frames, gaps = [], {}
        for y in years:
            df, missing = load_860_units(y)
            frames.append(df)
            gaps[y] = missing
            print(f"--- {y} ---")
            print(f"  battery generators : {len(df)}")
            print(f"  plants             : {df['Plant Code'].nunique()}")
            if missing:
                print(f"  COLUMNS NOT FOUND  : {missing}")
        return pd.concat(frames, ignore_index=True)[WANTED + ["Year"]], gaps


    units_all, column_gaps = build_units(YEARS)

    print("\nUnit table:", units_all.shape)
    print("Generators per plant-year:")
    print(units_all.groupby(["Year", "Plant Code"]).size().value_counts().to_string())
    units_all.head(8)
    return (units_all,)


@app.cell
def _(PROC, pd, units_all):
    # ===== attach the node label, restrict to WECC and to dispatchable =====
    # plant_to_node.csv is the authority for both questions: it was built from
    # the WECC-filtered panel and carries the nearest2 node assignment. Joining
    # on it keeps this table consistent with the two existing deliverables.
    # Status is applied here rather than downstream so that every later cell
    # consumes one table. EIA-860 Instructions Table 4 records availability,
    # not merely presence: SB is "available for service but not normally used"
    # and is dispatchable, whereas OS and OA describe units out of service for
    # the reporting year and are not. Summing the latter into a nodal bound
    # would let a dispatch model schedule capacity that cannot respond. RE, CN
    # and IP never reach this table; they appear only on the Retired and
    # Canceled sheet, which the loader does not read.
    #
    # units_wecc_all preserves the all-status population so that the
    # reconciliation below can still be run against the node capacity
    # deliverable, which was built before this filter existed.
    DISPATCHABLE = ("OP", "SB")

    plant_to_node = pd.read_csv(PROC / "plant_to_node.csv")

    units_wecc_all = units_all.merge(
        plant_to_node[["Plant Code", "geohash", "match_dist_km"]],
        on="Plant Code",
        how="inner",
    )
    units = (
        units_wecc_all[units_wecc_all["Status"].isin(DISPATCHABLE)]
        .reset_index(drop=True)
    )

    print("Units: all US", len(units_all),
          "-> WECC", len(units_wecc_all),
          "-> dispatchable", len(units))
    print("statuses removed:",
          units_wecc_all.loc[
              ~units_wecc_all["Status"].isin(DISPATCHABLE), "Status"
          ].value_counts().to_dict())
    print("Plants:", units["Plant Code"].nunique(),
          "| nodes:", units["geohash"].nunique())

    print("\nGenerators by year, dispatchable only:")
    print(units.groupby("Year").agg(
        generators=("Generator ID", "size"),
        plants=("Plant Code", "nunique"),
        nodes=("geohash", "nunique"),
    ).to_string())

    # A node whose every unit is non-dispatchable leaves the table rather than
    # appearing with a zero bound. A node silently absent from a deliverable is
    # harder to notice than one carrying zeros, so they are named.
    _lost = sorted(
        set(zip(units_wecc_all["geohash"], units_wecc_all["Year"]))
        - set(zip(units["geohash"], units["Year"]))
    )
    print("\nnode-years lost to the status filter:", len(_lost))
    for _g, _y in _lost:
        print(f"  {_g}  {_y}")
    return DISPATCHABLE, units, units_wecc_all


@app.cell
def _(DISPATCHABLE, units, units_wecc_all):
    # ===== generator status =====
    # Reported on the pre-filter population, since the point is to quantify
    # what the filter removed. OP operating, SB standby, OS out of service and
    # not returning, OA out of service and expected back.
    print("WECC generator-years by status, before filtering:")
    print(
        units_wecc_all.groupby(["Year", "Status"]).size()
        .unstack(fill_value=0).to_string()
    )

    _excluded = ~units_wecc_all["Status"].isin(DISPATCHABLE)
    print(f"\nexcluded as non-dispatchable:"
          f" {int(_excluded.sum())} / {len(units_wecc_all)}"
          f" ({_excluded.mean() * 100:.1f}%)")

    if _excluded.any():
        _mw = "Nameplate Capacity (MW)"
        _cap = units_wecc_all.groupby("Year").agg(
            MW_all=(_mw, "sum"),
        )
        _cap["MW_dispatchable"] = units.groupby("Year")[_mw].sum()
        _cap["MW_removed"] = _cap["MW_all"] - _cap["MW_dispatchable"]
        _cap["removed_%"] = _cap["MW_removed"] / _cap["MW_all"] * 100
        print("\ncapacity effect of the filter")
        print(_cap.round(2).to_string())
        print(
            "\nThe share is small in most years but not uniform across them,"
            "\nso the affected year should be named wherever a figure derived"
            "\nfrom these bounds is quoted."
        )
    return


@app.cell
def _(TECH_COLS, TECH_NAMES, units):
    # ===== field inventory: technology =====
    # A generator may declare up to four technologies, but here the results shows all the batteries in WECC from 2018-2022 have only one technology
    print("Storage Technology 1, generator counts by year:")
    print(units.groupby("Year")["Storage Technology 1"]
          .value_counts().unstack(fill_value=0).to_string())

    units_tech_count = units[TECH_COLS].notna().sum(axis=1)
    print("\nTechnologies declared per generator:")
    print(units_tech_count.value_counts().sort_index().to_string())

    print("\nCodes present, against Instructions Table 5b:")
    _seen = set()
    for _c in TECH_COLS:
        _seen |= set(units[_c].dropna().unique())
    for _code in sorted(_seen):
        print(f"  {_code}  {TECH_NAMES.get(_code, 'UNKNOWN CODE')}")

    print("\nLithium-ion share by year (count and capacity):")
    _lib = units["Storage Technology 1"] == "LIB"
    _by_year = units.groupby("Year").agg(
        generators=("Generator ID", "size"),
        MW=("Nameplate Capacity (MW)", "sum"),
    )
    _by_year["LIB_generators"] = units[_lib].groupby("Year").size()
    _by_year["LIB_MW"] = units[_lib].groupby("Year")["Nameplate Capacity (MW)"].sum()
    _by_year["LIB_count_%"] = (_by_year["LIB_generators"] / _by_year["generators"] * 100).round(1)
    _by_year["LIB_MW_%"] = (_by_year["LIB_MW"] / _by_year["MW"] * 100).round(1)
    print(_by_year.round(1).to_string())
    return


@app.cell
def _(ENCLOSURE_NAMES, units):
    # ===== field inventory: enclosure type =====
    print("Storage Enclosure Type, by year:")
    print(units.groupby("Year")["Storage Enclosure Type"]
          .value_counts(dropna=False).unstack(fill_value=0).to_string())
    print()
    for _code, _name in ENCLOSURE_NAMES.items():
        print(f"  {_code}  {_name}")
    return


@app.cell
def _(units):
    # ===== field inventory: reactive power and the max power denominator =====

    # Note the unit: EIA-860 Instructions line 38 specifies MVAR, reactive
    # power, not MVA apparent power. Under MVA, exceeding the real power rating
    # would be arithmetic; under MVAR it is a statement about the hardware.
    _q = units["Nameplate Reactive Power Rating"]
    _mx = units[["Maximum Charge Rate (MW)", "Maximum Discharge Rate (MW)"]].max(axis=1)
    _e = units["Nameplate Energy Capacity (MWh)"]

    print("input coverage across all WECC generator-years:", len(units))
    for _name, _s in [("reactive power (MVAR)", _q),
                      ("max(charge, discharge) MW", _mx),
                      ("energy capacity (MWh)", _e)]:
        print(f"  {_name:28s} missing {int(_s.isna().sum()):4d}"
              f" | zero {int((_s == 0).sum()):4d}")

    _usable = _q.notna() & _mx.notna() & _mx.gt(0) & _e.notna() & _e.gt(0)
    print(f"\nrows usable for an energy-weighted ratio: {int(_usable.sum())} / {len(units)}")

    print("\nasymmetric power ratings (charge != discharge):",
          int((units["Maximum Charge Rate (MW)"]
               != units["Maximum Discharge Rate (MW)"]).sum()))

    _exceeds = _q > _mx
    print(f"generators where reactive rating exceeds max power: {int(_exceeds.sum())}"
          f" ({_exceeds.mean() * 100:.1f}%)")
    return


@app.cell
def _(APP_COLS, pd, units):
    # ===== field inventory: the eleven application flags =====

    _rows = []
    for _c in APP_COLS:
        _v = units[_c].astype("string").str.strip().str.upper()
        _y = int((_v == "Y").sum())
        _n = int((_v == "N").sum())
        _rows.append({
            "application": _c,
            "Y": _y,
            "N": _n,
            "answered": _y + _n,
            "response_rate_%": round((_y + _n) / len(units) * 100, 1),
            "Y_share_%": round(_y / (_y + _n) * 100, 1) if (_y + _n) else None,
        })

    app_summary = pd.DataFrame(_rows).sort_values("response_rate_%", ascending=False)

    print(f"generator-years: {len(units)}")
    print(app_summary.to_string(index=False))

    print("\nVoltage or Reactive Power Support is the four-quadrant assumption check:")
    _vq = app_summary.loc[app_summary["application"]
                          == "Voltage or Reactive Power Support"].iloc[0]
    print(f"  {_vq['Y']} yes, {_vq['N']} no, {len(units) - _vq['answered']} blank")
    print(f"  {_vq['Y_share_%']}% of the {_vq['answered']} that answered,"
          f" but only {_vq['response_rate_%']}% answered at all")
    return


@app.cell
def _(DISPATCHABLE, PROC, units):
    # ===== save =====
    units.to_csv(PROC / "battery_units_2018_2022.csv", index=False)
    print("Saved: battery_units_2018_2022.csv")
    print("Rows:", len(units),
          "| plants:", units["Plant Code"].nunique(),
          "| nodes:", units["geohash"].nunique())
    print("Grain: one row per (Plant Code, Generator ID, Year)")
    print("Scope: Prime Mover BA, Operable sheet,"
          f" Status in {DISPATCHABLE}")
    return


if __name__ == "__main__":
    app.run()
