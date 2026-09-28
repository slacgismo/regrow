import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ==================================================================
    # tech_and_applications.py
    #
    # Storage technology mix and declared applications for the WECC battery
    # fleet, 2018-2022. Two things are settled here.
    #
    # 1. Per-node chemistry. The dispatch model configures one aggregate
    #    battery per node, so it needs to know what is at that node. Reported
    #    at two grains: the full breakdown, one row per node-year-technology,
    #    and a one-row-per-node-year summary derived from it.
    #
    # 2. Four-quadrant operation. The reactive support flag is the only field
    #    stating whether an installation is intended to provide voltage
    #    support. It is the one independent check on the four-quadrant
    #    assumption that reactive_power_ratio.py examines from the ratings.
    #
    # Weighting: the mix is reported under generator count, nameplate power and
    # nameplate energy. They disagree, and only the last two describe the
    # system; the count share is shown to demonstrate how far it misleads.
    #
    # Response rates: the application fields are voluntary and mostly blank, so
    # every proportion is reported beside the response rate that produced it.
    #
    # INPUT
    #     battery_units_2018_2022.csv        generator-level unit table
    #
    # OUTPUT
    #     technology_by_node_year.csv        node x year x technology, long
    #     technology_node_mixing.csv         node x year summary of the above
    #     technology_mix_by_year.csv         fleet mix under three denominators
    #     applications_summary.csv           flags with response rates
    #     applications_by_year.csv           flags by year
    #     fig_technology_mix.png             count against capacity, by year
    #     fig_lithium_share.png              share under each denominator
    #     fig_application_response.png       response rate against yes-share
    #     fig_reactive_application.png       reactive support, per year
    #
    # marimo: each name is bound in exactly one cell; cell-local names are
    # underscore prefixed.
    # ==================================================================
    return


@app.cell
def _():
    # ===== imports, input, parameters =====
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pathlib import Path
    import marimo as mo

    # Paths resolve from the notebook's own location, so this runs from a fresh
    # clone regardless of the working directory marimo was started in.
    _nb_dir = mo.notebook_dir()
    if _nb_dir is None:                        # running as a plain script
        _nb_dir = Path(globals().get("__file__", ".")).resolve().parent
    _BASE = Path(_nb_dir).resolve().parent     # data/batteries/
    PROC = _BASE / "processed"
    PROC.mkdir(parents=True, exist_ok=True)

    units = pd.read_csv(PROC / "battery_units_2018_2022.csv")

    MW_COL = "Nameplate Capacity (MW)"
    E_COL = "Nameplate Energy Capacity (MWh)"
    Q_COL = "Nameplate Reactive Power Rating"

    # EIA-860 Instructions, Schedule 3, Table 5b. Must stay identical to the
    # copy in build_unit_table.py. The two drifted apart once, and the copy
    # here was a revision behind, so NAB and PBB passed through unlabelled.
    # Codes present 2018-2022: LIB 417, NAB 12, PBB 6, FLB 5.
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

    # The chemistry the fleet mix is read against: shares are reported for it
    # and for the remainder. A reference point, not an assumption.
    REFERENCE_TECH = "Lithium-ion battery"

    print("unit table:", units.shape)
    print(
        "generator-years:", len(units),
        "| plants:", units["Plant Code"].nunique(),
        "| nodes:", units["geohash"].nunique(),
    )
    return (
        ENCLOSURE_NAMES,
        E_COL,
        MW_COL,
        PROC,
        Q_COL,
        REFERENCE_TECH,
        TECH_NAMES,
        pd,
        plt,
        units,
    )


@app.cell
def _(units):
    # ===== column discovery =====
    # EIA column names move between vintages, so fields are located by pattern
    # and the result printed. A silent mismatch would produce empty tables
    # rather than an error, which is the failure mode this cell prevents.
    #
    # Technology columns are ordered by trailing index, not file position,
    # because the first is taken as the primary declaration.
    def _tech_rank(name):
        _digits = "".join(ch for ch in name if ch.isdigit())
        return int(_digits) if _digits else 0

    TECH_COLS = sorted(
        (c for c in units.columns if c.lower().startswith("storage technology")),
        key=_tech_rank,
    )
    ENC_COL = next((c for c in units.columns if "enclosure" in c.lower()), None)

    _CANDIDATES = {
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
        "Storage Applications Other",
    }
    APP_COLS = [c for c in units.columns if c in _CANDIDATES]
    if not APP_COLS:
        # Fall back on shape: a flag column holds only Y, N or blank. Fields
        # already identified are excluded, since a technology or enclosure
        # column could satisfy that test by accident.
        _taken = set(TECH_COLS) | ({ENC_COL} if ENC_COL else set())
        APP_COLS = [
            c
            for c in units.columns
            if c not in _taken
            and units[c].notna().any()
            and units[c]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(["Y", "N"])
            .all()
        ]
        print("candidate names did not match; flags identified by shape instead.")

    VQ_COL = next(
        (c for c in APP_COLS if "reactive" in c.lower() or "voltage" in c.lower()),
        None,
    )

    print("technology columns :", TECH_COLS if TECH_COLS else "NONE FOUND")
    print("enclosure column   :", ENC_COL)
    print("application columns:", len(APP_COLS))
    for _c in APP_COLS:
        print("   ", _c)
    print("reactive support   :", VQ_COL)
    if not TECH_COLS or not APP_COLS:
        print("\nA field could not be located. Inspect units.columns before"
              " reading anything below.")
    return APP_COLS, ENC_COL, TECH_COLS, VQ_COL


@app.cell
def _(E_COL, MW_COL, TECH_COLS, TECH_NAMES, pd, units):
    # ===== technology assignment =====
    # The form allows up to four technologies per generator. The first field is
    # the primary declaration and carries the mix; generators declaring more
    # than one are reported separately, because a unit split across chemistries
    # would need an apportionment rule and the mix would stop being a partition.
    #
    # Multiplicity counts distinct codes, not populated cells: the same
    # chemistry named twice is one chemistry.
    #
    # Coverage is reported in energy as well as units, since everything
    # downstream is computed over the declared subset.
    tech = units.copy()
    _primary = TECH_COLS[0] if TECH_COLS else None

    def _clean(series):
        _s = series.astype("string").str.strip().str.upper()
        return _s.replace("", None)

    _codes_by_col = pd.DataFrame({c: _clean(tech[c]) for c in TECH_COLS})

    tech["technology"] = (
        _codes_by_col[_primary]
        .map(lambda v: TECH_NAMES.get(v, v))
        .astype("object")
        .where(_codes_by_col[_primary].notna())
    )
    tech["n_technologies"] = _codes_by_col.nunique(axis=1)

    _declared = tech["technology"].notna()
    _codes = sorted(set(_codes_by_col[_primary].dropna()))
    _unknown = [c for c in _codes if c not in TECH_NAMES]

    print("primary technology field:", _primary)
    print("codes present:", _codes)
    if _unknown:
        print("  NOT IN THE CODE TABLE:", _unknown)
        print("  These pass through under their own code and appear unlabelled")
        print("  in every table and figure below.")

    print(
        "\ngenerator-years with a technology declared:",
        int(_declared.sum()), "of", len(tech),
        f"({_declared.mean() * 100:.1f}%)",
    )
    for _label, _col in [("power, MW", MW_COL), ("energy, MWh", E_COL)]:
        _tot = tech[_col].sum()
        _cov = tech.loc[_declared, _col].sum()
        print(f"  {_label:<12} covered: {_cov:,.1f} of {_tot:,.1f}"
              f" ({_cov / _tot * 100:.2f}%)" if _tot else f"  {_label}: no data")

    print(
        "\ngenerator-years declaring more than one distinct technology:",
        int((tech["n_technologies"] > 1).sum()),
    )
    if int((tech["n_technologies"] > 1).sum()) == 0:
        print("  None. The mix is a partition and no apportionment rule between"
              "\n  co-declared chemistries is required.")
    else:
        print(
            tech.loc[tech["n_technologies"] > 1,
                     ["Year", "Plant Code", "Plant Name"] + TECH_COLS]
            .to_string(index=False)
        )
    return (tech,)


@app.cell
def _(E_COL, MW_COL, tech):
    # ===== fleet mix under three denominators =====
    # Reported together because they disagree, and the disagreement determines
    # which may be quoted. Shares are taken over the declared subset, so each
    # year sums to 100 by construction; the size of that subset is reported in
    # the cell above, not implied here.
    _t = tech[tech["technology"].notna()]

    mix = (
        _t.groupby(["Year", "technology"])
        .agg(generators=("technology", "size"),
             MW=(MW_COL, "sum"),
             MWh=(E_COL, "sum"))
        .reset_index()
    )
    for _num, _name in [
        ("generators", "unit_share_%"),
        ("MW", "MW_share_%"),
        ("MWh", "MWh_share_%"),
    ]:
        mix[_name] = (
            mix.groupby("Year")[_num].transform(lambda s: s / s.sum() * 100).round(2)
        )
    mix[["MW", "MWh"]] = mix[["MW", "MWh"]].round(1)

    print("technology mix by year")
    print(mix.to_string(index=False))
    return (mix,)


@app.cell
def _(REFERENCE_TECH, mix):
    # ===== share of the reference chemistry =====
    # The size of the remainder after the dominant chemistry is the operative
    # number: it says how far the fleet can be described by one chemistry.
    #
    # Reindexed over every year in the mix: without it, a year in which the
    # reference chemistry is absent would drop out silently and every
    # downstream .loc would read the wrong row.
    _years = sorted(mix["Year"].unique())
    lib_share = (
        mix[mix["technology"] == REFERENCE_TECH]
        .set_index("Year")[["generators", "unit_share_%",
                            "MW_share_%", "MWh_share_%"]]
        .reindex(_years)
        .fillna(0.0)
    )
    lib_share["generators"] = lib_share["generators"].astype(int)
    lib_share.index.name = "Year"
    for _c in ["unit", "MW", "MWh"]:
        lib_share[f"non_{_c}_share_%"] = (100 - lib_share[f"{_c}_share_%"]).round(2)

    print(f"{REFERENCE_TECH} share by year")
    print(lib_share.to_string())

    if (lib_share["generators"] == 0).any():
        print("\nYears in which the reference chemistry is absent:",
              list(lib_share.index[lib_share["generators"] == 0]))

    _last = lib_share.index.max()
    print(
        f"\nIn {_last} the remainder is"
        f" {lib_share.loc[_last, 'non_unit_share_%']:.2f}% of generators,"
        f" {lib_share.loc[_last, 'non_MW_share_%']:.2f}% of power and"
        f" {lib_share.loc[_last, 'non_MWh_share_%']:.2f}% of energy."
    )
    return (lib_share,)


@app.cell
def _(E_COL, MW_COL, tech):
    # ===== technology by node and year: the full breakdown =====
    # One row per (Year, geohash, technology). This is the source of truth for
    # everything per-node below; the summary in the next cell is a rollup of
    # it, so the two cannot disagree.
    #
    # Long rather than wide: the set of technologies present changes between
    # years, and a wide table would carry a column of zeros for every chemistry
    # absent from a given node-year. A consumer that wants a matrix can pivot
    # on the three keys.
    #
    # share_MW and share_MWh are within the node-year, so each node-year sums
    # to 1 across its technologies. They are the quantity that says whether a
    # single-chemistry label describes the node: a node at 0.99 and a node at
    # 0.51 are different facts.
    _t = tech[tech["technology"].notna()]

    node_tech_long = (
        _t.groupby(["Year", "geohash", "technology"])
        .agg(generators=("technology", "size"),
             nameplate_MW=(MW_COL, "sum"),
             energy_MWh=(E_COL, "sum"))
        .reset_index()
    )

    # Guarded: a node-year reporting no capacity would divide by zero, and
    # power and energy are guarded separately because a node can report MW
    # while leaving MWh blank.
    _totals = node_tech_long.groupby(["Year", "geohash"])[
        ["nameplate_MW", "energy_MWh"]
    ].transform("sum")
    node_tech_long["share_MW"] = (
        node_tech_long["nameplate_MW"] / _totals["nameplate_MW"]
    ).where(_totals["nameplate_MW"] > 0).round(4)
    node_tech_long["share_MWh"] = (
        node_tech_long["energy_MWh"] / _totals["energy_MWh"]
    ).where(_totals["energy_MWh"] > 0).round(4)

    node_tech_long[["nameplate_MW", "energy_MWh"]] = node_tech_long[
        ["nameplate_MW", "energy_MWh"]
    ].round(1)
    node_tech_long = (
        node_tech_long
        .sort_values(["Year", "geohash", "nameplate_MW"],
                     ascending=[True, True, False])
        .reset_index(drop=True)
    )

    print("technology by node and year:", len(node_tech_long), "rows |",
          node_tech_long["geohash"].nunique(), "nodes |",
          node_tech_long.groupby(["Year", "geohash"]).ngroups, "node-years")
    print("\nrows per technology")
    print(node_tech_long["technology"].value_counts().to_string())
    print("\nnameplate MW by technology and year")
    print(
        node_tech_long.pivot_table(index="Year", columns="technology",
                                   values="nameplate_MW", aggfunc="sum")
        .fillna(0).round(1).to_string()
    )
    return (node_tech_long,)


@app.cell
def _(node_tech_long):
    # ===== node-year summary, rolled up from the breakdown =====
    # One row per node-year for consumers that want a single label per node.
    # Every column is computed from node_tech_long, so the summary cannot drift
    # away from the breakdown it summarises.
    #
    # technology_dominant is decided by nameplate MW, not generator count,
    # because the model acts on capacity: a node can hold one large unit of one
    # chemistry beside several small units of another. Ties break
    # alphabetically so the column is stable across runs.
    _totals = (
        node_tech_long.groupby(["Year", "geohash"])
        .agg(technologies=("technology", "nunique"),
             generators=("generators", "sum"),
             nameplate_MW=("nameplate_MW", "sum"),
             energy_MWh=("energy_MWh", "sum"))
        .reset_index()
    )

    _dominant = (
        node_tech_long
        .sort_values(["nameplate_MW", "technology"], ascending=[False, True])
        .drop_duplicates(subset=["Year", "geohash"], keep="first")
        .rename(columns={"technology": "technology_dominant",
                         "nameplate_MW": "dominant_MW",
                         "share_MW": "dominant_share_MW"})
    )

    node_tech = _totals.merge(
        _dominant[["Year", "geohash", "technology_dominant",
                   "dominant_MW", "dominant_share_MW"]],
        on=["Year", "geohash"],
        how="left",
    )
    node_tech["mixed"] = node_tech["technologies"] > 1
    node_tech[["nameplate_MW", "energy_MWh"]] = node_tech[
        ["nameplate_MW", "energy_MWh"]
    ].round(1)

    node_tech = node_tech[
        ["Year", "geohash", "technology_dominant", "dominant_share_MW", "mixed",
         "technologies", "generators", "nameplate_MW", "energy_MWh",
         "dominant_MW"]
    ].sort_values(["Year", "geohash"]).reset_index(drop=True)

    _n_mixed = int(node_tech["mixed"].sum())
    _mw_total = node_tech["nameplate_MW"].sum()
    _mw_mixed = node_tech.loc[node_tech["mixed"], "nameplate_MW"].sum()

    # The rollup must reproduce the breakdown exactly; if it does not, one of
    # the two aggregations is wrong and every per-node number is suspect.
    _delta = abs(node_tech["nameplate_MW"].sum()
                 - node_tech_long["nameplate_MW"].sum())
    if _delta > 0.05:
        print(f"SUMMARY DOES NOT RECONCILE with the breakdown: {_delta:.3f} MW")

    print("node-years:", len(node_tech),
          "| nodes:", node_tech["geohash"].nunique())
    print("\ndominant chemistry by node-year")
    print(node_tech["technology_dominant"].value_counts().to_string())
    print("\nnode-years with more than one technology:", _n_mixed,
          "of", len(node_tech))
    print(f"capacity at mixed nodes: {_mw_mixed:,.1f} of {_mw_total:,.1f} MW"
          f" ({_mw_mixed / _mw_total * 100:.2f}%)" if _mw_total else "no capacity")

    if _n_mixed:
        print("\nmixed node-years, largest first")
        print(
            node_tech[node_tech["mixed"]]
            .sort_values(["Year", "nameplate_MW"], ascending=[False, False])
            .head(15)
            .to_string(index=False)
        )
        print("\nAt these nodes technology_dominant is a majority label, not a"
              "\ncomplete description. The constituents are in"
              "\ntechnology_by_node_year.csv.")
    else:
        print("  Every node is single-technology; the label is exact.")
    return (node_tech,)


@app.cell
def _(ENCLOSURE_NAMES, ENC_COL, MW_COL, units):
    # ===== enclosure type =====
    # Not a model parameter. Retained because the category distribution shifts
    # between vintages in a way that looks like a reporting change rather than
    # a fleet change. Counts and capacity are shown together: a category that
    # is a third of the units and a twentieth of the megawatts is a different
    # fact than the count alone suggests.
    #
    # Unknown codes are reported rather than passed through silently, which is
    # how CS - the largest category - went unlabelled in an earlier revision.
    if ENC_COL is None:
        print("no enclosure column; skipped.")
    else:
        _e = units[units[ENC_COL].notna()].copy()
        _raw = _e[ENC_COL].astype(str).str.strip().str.upper()

        _unknown_enc = sorted(set(_raw) - set(ENCLOSURE_NAMES))
        if _unknown_enc:
            print("ENCLOSURE CODES NOT IN THE TABLE:", _unknown_enc)

        _e["enclosure"] = _raw.map(lambda v: ENCLOSURE_NAMES.get(v, v))
        print("generator-years by enclosure type")
        print(
            _e.pivot_table(index="Year", columns="enclosure",
                           values=MW_COL, aggfunc="size")
            .fillna(0).astype(int).to_string()
        )
        print("\nnameplate MW by enclosure type")
        print(
            _e.pivot_table(index="Year", columns="enclosure",
                           values=MW_COL, aggfunc="sum")
            .fillna(0).round(1).to_string()
        )
        print("\nblank enclosure cells:",
              int(units[ENC_COL].isna().sum()), "of", len(units))
    return


@app.cell
def _(APP_COLS, pd, units):
    # ===== application flags, normalised once =====
    # Every downstream table derives from this frame. The parse used to be
    # repeated in three cells with three treatments of the edge cases, which is
    # exactly the arrangement in which treatments drift apart.
    #
    # Four responses are distinguished, not two. A cell that is neither Y nor N
    # nor empty is a reporting anomaly and is counted as such; folding it into
    # the blank count would inflate apparent non-response and hide the anomaly
    # at the same time.
    _long = units[["Year"] + APP_COLS].melt(
        id_vars="Year", var_name="application", value_name="raw"
    )
    _v = _long["raw"].astype("string").str.strip().str.upper().replace("", pd.NA)

    flags = _long[["Year", "application"]].copy()
    flags["response"] = "other"
    flags.loc[_v.isna().to_numpy(), "response"] = "blank"
    flags.loc[_v.eq("Y").fillna(False).to_numpy(), "response"] = "Y"
    flags.loc[_v.eq("N").fillna(False).to_numpy(), "response"] = "N"

    def tally(frame, keys):
        _g = (
            frame.groupby(keys)["response"]
            .value_counts()
            .unstack(fill_value=0)
        )
        for _c in ["Y", "N", "blank", "other"]:
            if _c not in _g.columns:
                _g[_c] = 0
        _g = _g[["Y", "N", "blank", "other"]]
        _g["total"] = _g.sum(axis=1)
        _g["answered"] = _g["Y"] + _g["N"]
        _g["response_rate_%"] = (_g["answered"] / _g["total"] * 100).round(1)
        _g["Y_share_of_answered_%"] = (
            (_g["Y"] / _g["answered"] * 100).round(1).where(_g["answered"] > 0)
        )
        _g["Y_share_of_fleet_%"] = (_g["Y"] / _g["total"] * 100).round(1)
        return _g

    _anom = int((flags["response"] == "other").sum())
    print("flag cells parsed:", len(flags))
    print("cells that are neither Y, N nor empty:", _anom)
    if _anom:
        print(
            flags[flags["response"] == "other"]
            .groupby("application").size().to_string()
        )
    return flags, tally


@app.cell
def _(flags, tally):
    # ===== declared applications, pooled =====
    # Three proportions per field, answering different questions.
    # Y_share_of_answered describes respondents; Y_share_of_fleet describes the
    # fleet; response_rate says how far apart those populations are. Quoting
    # the first without the third overstates the evidence by the reciprocal of
    # the response rate.
    apps = (
        tally(flags, ["application"])
        .sort_values("response_rate_%", ascending=False)
        .reset_index()
    )

    print("declared applications, ordered by response rate")
    print(apps.to_string(index=False))
    return (apps,)


@app.cell
def _(flags, tally):
    # ===== applications by year =====
    # Response rates move over time. A field answered by a tenth of the fleet
    # in one vintage and half in another does not support a pooled statement.
    apps_by_year = tally(flags, ["application", "Year"]).reset_index()

    print("response rate by application and year, percent")
    print(
        apps_by_year.pivot(
            index="application", columns="Year", values="response_rate_%"
        ).to_string()
    )
    return (apps_by_year,)


@app.cell
def _(VQ_COL, apps_by_year, pd):
    # ===== reactive support, by year =====
    # Split out of the pooled table rather than recomputed, so the two cannot
    # disagree. This is the field bearing on the four-quadrant assumption.
    if VQ_COL is None:
        vq_by_year = pd.DataFrame()
        print("no reactive support column; skipped.")
    else:
        vq_by_year = (
            apps_by_year[apps_by_year["application"] == VQ_COL]
            .set_index("Year")
            .drop(columns="application")
        )
        print(f"{VQ_COL}, by year")
        print(vq_by_year.to_string())
    return (vq_by_year,)


@app.cell
def _(Q_COL, VQ_COL, pd, units):
    # ===== reactive support: stated intent against reported rating =====
    # The flag and the nameplate reactive rating are independent fields. Units
    # declining the application while reporting a non-zero rating, or accepting
    # it while reporting zero, show the two are populated independently, which
    # caps how far either can be relied on.
    if VQ_COL is None:
        vq_disagree = 0
        print("no reactive support column; skipped.")
    else:
        _v = units[VQ_COL].astype("string").str.strip().str.upper().fillna("blank")
        _q = units[Q_COL]
        _qclass = pd.Series("Q absent", index=units.index, dtype="object")
        _qclass[_q.eq(0)] = "Q = 0"
        _qclass[_q.gt(0)] = "Q > 0"
        _cross = pd.crosstab(_v, _qclass)
        print("stated application against the reactive rating reported")
        print(_cross.to_string())

        vq_disagree = 0
        if "N" in _cross.index and "Q > 0" in _cross.columns:
            vq_disagree += int(_cross.loc["N", "Q > 0"])
        if "Y" in _cross.index and "Q = 0" in _cross.columns:
            vq_disagree += int(_cross.loc["Y", "Q = 0"])
        print(
            f"\ngenerator-years where the two fields disagree: {vq_disagree}"
            "\nThe ratings are the better evidence on capability: they are"
            "\nreported for almost the whole fleet, whereas this flag is not."
        )
    return (vq_disagree,)


@app.cell
def _(REFERENCE_TECH, mix, plt):
    # ===== figure 1: mix by count, power and energy =====
    # Three denominators side by side because the contrast is the finding:
    # reading the count panel alone overstates how much of the fleet is not the
    # dominant chemistry.
    #
    # The lower row repeats the upper with the dominant chemistry removed and
    # the axis rescaled, because at these shares the remainder is a few pixels
    # tall at full scale.
    _views = [
        ("unit_share_%", "share of generators"),
        ("MW_share_%", "share of power, MW"),
        ("MWh_share_%", "share of energy, MWh"),
    ]
    _cols = sorted(mix["technology"].unique())
    _others = [c for c in _cols if c != REFERENCE_TECH]
    _palette = ["#3c6e88", "#b4442e", "#6f9b6e", "#c8a15a",
                "#8a6f9b", "#7fa8bd", "#999999"]
    _colour = {c: _palette[i % len(_palette)] for i, c in enumerate(_cols)}

    fig_mix, _axes = plt.subplots(2, 3, figsize=(13, 7.6))

    def _stack(ax, piv, order):
        _bottom = [0.0] * len(piv)
        for _c in order:
            if _c not in piv.columns:
                continue
            _v = list(piv[_c].values)
            ax.bar([str(_y) for _y in piv.index], _v, bottom=_bottom,
                   color=_colour[_c], label=_c, width=0.62)
            _bottom = [_b + _x for _b, _x in zip(_bottom, _v)]
        ax.grid(alpha=0.2, axis="y")
        ax.tick_params(labelsize=8)

    _pivots = {
        _col: mix.pivot(index="Year", columns="technology", values=_col).fillna(0)
        for _col, _ in _views
    }
    _headroom = max(
        (_p[_others].sum(axis=1).max() for _p in _pivots.values()), default=0.0
    ) if _others else 0.0

    for _j, (_col, _title) in enumerate(_views):
        _stack(_axes[0, _j], _pivots[_col], _cols)
        _axes[0, _j].set_ylim(0, 100)
        _axes[0, _j].set_title(_title, fontsize=10)

        _stack(_axes[1, _j], _pivots[_col], _others)
        _axes[1, _j].set_ylim(0, max(_headroom * 1.25, 1.0))
        _axes[1, _j].set_title(f"{_title}, excluding {REFERENCE_TECH}", fontsize=9)

    _axes[0, 0].set_ylabel("percent")
    _axes[1, 0].set_ylabel("percent")
    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    fig_mix.legend(_handles, _labels, fontsize=8, frameon=False,
                   loc="lower center", ncol=max(len(_cols), 1))
    fig_mix.suptitle(
        "Technology mix under three denominators; lower row rescaled to the"
        f" remainder after {REFERENCE_TECH}",
        fontsize=11,
    )
    fig_mix.tight_layout(rect=(0, 0.05, 1, 1))
    fig_mix
    return (fig_mix,)


@app.cell
def _(REFERENCE_TECH, lib_share, plt):
    # ===== figure 2: reference share under each denominator =====
    # One line per denominator, on one axis, because the gap between them is
    # the point: the count share understates the dominance that the capacity
    # and energy shares report.
    fig_lib, _ax = plt.subplots(figsize=(6.4, 4.2))

    for _c, _m, _col in [
        ("unit_share_%", "o", "#999999"),
        ("MW_share_%", "s", "#3c6e88"),
        ("MWh_share_%", "^", "#b4442e"),
    ]:
        _ax.plot(lib_share.index, lib_share[_c], marker=_m, lw=1.8, ms=6,
                 color=_col, label=_c)

    _ax.set_ylim(min(60, lib_share[["unit_share_%", "MW_share_%",
                                    "MWh_share_%"]].min().min() - 5), 101)
    _ax.set_xticks(list(lib_share.index))
    _ax.set_ylabel("percent")
    _ax.set_title(f"{REFERENCE_TECH} share by denominator", fontsize=10)
    _ax.legend(fontsize=8, frameon=False, loc="lower right")
    _ax.grid(alpha=0.25)

    fig_lib.tight_layout()
    fig_lib
    return (fig_lib,)


@app.cell
def _(apps, plt):
    # ===== figure 3: response rate against yes-share =====
    # The gap between the bars is the point: a long yes-share beside a short
    # response rate is an application that looks universal and was answered by
    # almost nobody.
    _o = apps.sort_values("response_rate_%")
    _y = list(range(len(_o)))

    fig_apps, _ax = plt.subplots(figsize=(9.5, 5.4))
    _ax.barh([_v - 0.2 for _v in _y], _o["response_rate_%"], height=0.38,
             color="#3c6e88", label="response rate")
    _ax.barh([_v + 0.2 for _v in _y], _o["Y_share_of_answered_%"].fillna(0),
             height=0.38, color="#b4442e", label="yes share of respondents")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_o["application"], fontsize=8)
    _ax.set_xlabel("percent")
    _ax.set_xlim(0, 100)
    _ax.set_title("A high yes-share means little where the response rate is low",
                  fontsize=11)
    _ax.legend(fontsize=8, frameon=False, loc="lower right")
    _ax.grid(alpha=0.2, axis="x")
    fig_apps.tight_layout()
    fig_apps
    return (fig_apps,)


@app.cell
def _(VQ_COL, plt, vq_by_year):
    # ===== figure 4: reactive support, per year and pooled =====
    # Separated from the other flags because it bears on the four-quadrant
    # assumption. Both series share one axis so a high yes-share cannot be read
    # without the response rate beneath it.
    if VQ_COL is None or len(vq_by_year) == 0:
        fig_vq = None
        print("skipped.")
    else:
        _years = list(vq_by_year.index)
        _ncol = 3
        _nrow = -(-(len(_years) + 1) // _ncol)
        fig_vq, _axes = plt.subplots(_nrow, _ncol, figsize=(12, 3.2 * _nrow),
                                     sharey=True, squeeze=False)
        _flat = _axes.ravel()
        _labels = ["response\nrate", "yes share of\nrespondents"]

        for _i, _y in enumerate(_years):
            _r = vq_by_year.loc[_y]
            _ax = _flat[_i]
            _share = _r["Y_share_of_answered_%"]
            _ax.bar(_labels,
                    [_r["response_rate_%"], 0.0 if _share != _share else _share],
                    color=["#3c6e88", "#b4442e"], width=0.55)
            _ax.set_ylim(0, 100)
            _ax.set_title(
                f"{_y}   {int(_r['answered'])} of {int(_r['total'])} answered",
                fontsize=9,
            )
            _ax.grid(alpha=0.2, axis="y")
            _ax.tick_params(labelsize=8)

        _ax = _flat[len(_years)]
        _tot = int(vq_by_year["total"].sum())
        _ans = int(vq_by_year["answered"].sum())
        _yes = int(vq_by_year["Y"].sum())
        _ax.bar(_labels,
                [_ans / _tot * 100 if _tot else 0.0,
                 _yes / _ans * 100 if _ans else 0.0],
                color=["#3c6e88", "#b4442e"], width=0.55)
        _ax.set_ylim(0, 100)
        _ax.set_title(f"all years   {_ans} of {_tot} answered", fontsize=9)
        _ax.grid(alpha=0.2, axis="y")

        for _j in range(len(_years) + 1, len(_flat)):
            _flat[_j].axis("off")
        _flat[0].set_ylabel("percent")
        fig_vq.suptitle(f"{VQ_COL}: intent is stated by a minority", fontsize=11)
        fig_vq.tight_layout()
    fig_vq
    return (fig_vq,)


@app.cell
def _(
    PROC,
    apps,
    apps_by_year,
    fig_apps,
    fig_lib,
    fig_mix,
    fig_vq,
    mix,
    node_tech,
    node_tech_long,
):
    # ===== outputs =====
    node_tech_long.to_csv(PROC / "technology_by_node_year.csv", index=False)
    node_tech.to_csv(PROC / "technology_node_mixing.csv", index=False)
    mix.to_csv(PROC / "technology_mix_by_year.csv", index=False)
    apps.to_csv(PROC / "applications_summary.csv", index=False)
    apps_by_year.to_csv(PROC / "applications_by_year.csv", index=False)

    fig_mix.savefig(PROC / "fig_technology_mix.png", dpi=150, bbox_inches="tight")
    fig_lib.savefig(PROC / "fig_lithium_share.png", dpi=150, bbox_inches="tight")
    fig_apps.savefig(
        PROC / "fig_application_response.png", dpi=150, bbox_inches="tight"
    )
    if fig_vq is not None:
        fig_vq.savefig(
            PROC / "fig_reactive_application.png", dpi=150, bbox_inches="tight"
        )

    print("written")
    print(f"  technology_by_node_year.csv       {len(node_tech_long):5d} rows")
    print(f"  technology_node_mixing.csv        {len(node_tech):5d} rows")
    print(f"  technology_mix_by_year.csv        {len(mix):5d} rows")
    print(f"  applications_summary.csv          {len(apps):5d} rows")
    print(f"  applications_by_year.csv          {len(apps_by_year):5d} rows")
    print(f"  figures                           {3 + (fig_vq is not None)}")
    print()
    print("Model inputs from this notebook")
    print("  technology_by_node_year.csv, per node, year and technology:")
    print("    nameplate_MW, energy_MWh   capacity of that chemistry at that node")
    print("    share_MW, share_MWh        its share within the node-year")
    print("  technology_node_mixing.csv, per node and year:")
    print("    technology_dominant        chemistry holding the most MW")
    print("    dominant_share_MW          the share it holds")
    print("    mixed                      True where one chemistry is not enough")
    return


@app.cell
def _(
    E_COL,
    Q_COL,
    REFERENCE_TECH,
    VQ_COL,
    apps,
    lib_share,
    node_tech,
    node_tech_long,
    tech,
    units,
    vq_by_year,
    vq_disagree,
):
    # ===== findings =====
    # Every number is read from the current run and every comparative claim is
    # decided by a test on those numbers, so a sentence that would be false
    # under a different input is not printed under that input.
    _first = int(lib_share.index.min())
    _last = int(lib_share.index.max())

    print("=" * 70)
    print(f"FINDINGS   technology and applications, {_first}-{_last}")
    print("=" * 70)

    print(f"\n1. Composition. {REFERENCE_TECH} share in {_last}:")
    print(f"   by generator count {lib_share.loc[_last, 'unit_share_%']:.2f}%")
    print(f"   by power           {lib_share.loc[_last, 'MW_share_%']:.2f}%")
    print(f"   by energy          {lib_share.loc[_last, 'MWh_share_%']:.2f}%")
    _gap = lib_share.loc[_last, "MWh_share_%"] - lib_share.loc[_last, "unit_share_%"]
    if abs(_gap) >= 1:
        print(f"   The denominators differ by {_gap:+.2f} points; the count"
              " share is not usable as a weight.")
    else:
        print("   The denominators agree; either may be quoted.")

    print("\n2. Coverage of the declared subset.")
    _dec = tech["technology"].notna()
    _cov_units = _dec.mean() * 100
    _cov_energy = (
        tech.loc[_dec, E_COL].sum() / tech[E_COL].sum() * 100
        if tech[E_COL].sum() else float("nan")
    )
    print(f"   generator-years declaring a chemistry: {_cov_units:.1f}%")
    print(f"   the energy they hold                 : {_cov_energy:.2f}%")
    if _cov_energy >= 99:
        print("   The declared subset is the fleet in all but name, so the mix")
        print("   may be quoted as a fleet description without qualification.")
    else:
        print("   A material share of fleet energy carries no declared chemistry.")
        print("   The mix describes the declared subset and inherits whatever")
        print("   the undeclared remainder turns out to be.")

    print("\n3. Per-node chemistry, for model specification.")
    _multi = int((tech["n_technologies"] > 1).sum())
    _nmix = int(node_tech["mixed"].sum())
    _mw_mixed = node_tech.loc[node_tech["mixed"], "nameplate_MW"].sum()
    _mw_all = node_tech["nameplate_MW"].sum()
    _mw_share = _mw_mixed / _mw_all * 100 if _mw_all else float("nan")
    _ref_nodes = int((node_tech["technology_dominant"] == REFERENCE_TECH).sum())
    _min_share = node_tech["dominant_share_MW"].min()
    print(f"   node-year x technology rows: {len(node_tech_long)}")
    print(f"   node-years {len(node_tech)}, of which {_ref_nodes} are dominated"
          f" by {REFERENCE_TECH}")
    print(f"   lowest dominant share at any node-year: {_min_share:.4f}")
    print(f"   generator-years declaring more than one chemistry: {_multi}")
    print(f"   chemically mixed node-years: {_nmix} of {len(node_tech)}")
    print(f"   capacity behind them       : {_mw_share:.2f}% of node capacity")
    if _multi == 0:
        print("   No generator is split, so the mix is a partition and no")
        print("   apportionment rule between co-declared chemistries is needed.")
    else:
        print("   Some generators are split across chemistries; the mix is not a")
        print("   partition and an apportionment rule is required before the")
        print("   shares above can be read as exclusive.")
    if _nmix == 0:
        print("   Every node is single-technology: technology_dominant is an")
        print("   exact description and the breakdown adds nothing.")
    elif _mw_share < 5:
        print("   Mixed nodes exist but hold little capacity, so")
        print("   technology_dominant is a close description of every node;")
        print("   the breakdown is where the remainder can be read off.")
    else:
        print("   Mixed nodes hold a substantial share of capacity, so the")
        print("   breakdown, not the dominant label, is what the model needs.")

    print("\n4. Application fields are sparsely answered.")
    print(f"   response rate ranges {apps['response_rate_%'].min():.1f}%"
          f" to {apps['response_rate_%'].max():.1f}%")
    print(f"   median response rate {apps['response_rate_%'].median():.1f}%")
    _worst = apps.loc[apps["response_rate_%"].idxmin()]
    _ws = _worst["Y_share_of_answered_%"]
    print(f"   least answered: {_worst['application']}"
          f" at {_worst['response_rate_%']:.1f}%, yes-share among respondents"
          f" {'n/a' if _ws != _ws else f'{_ws:.1f}%'}")
    _anom = int(apps["other"].sum())
    if _anom:
        print(f"   cells that are neither Y, N nor empty: {_anom}")

    if VQ_COL is not None and len(vq_by_year):
        _tot = int(vq_by_year["total"].sum())
        _ans = int(vq_by_year["answered"].sum())
        _yes = int(vq_by_year["Y"].sum())
        print("\n5. Four-quadrant operation cannot be settled by the flag.")
        print(f"   {VQ_COL}")
        print(f"   answered {_ans} of {_tot} generator-years"
              f" ({_ans / _tot * 100:.1f}%)" if _tot else "   no records")
        if _ans:
            print(f"   yes among respondents {_yes / _ans * 100:.1f}%,"
                  f" yes as a share of the fleet {_yes / _tot * 100:.1f}%")
        _q = units[Q_COL]
        print(f"   the reactive rating, by contrast, is reported for"
              f" {int(_q.notna().sum())} of {len(units)}"
              f" ({_q.notna().mean() * 100:.1f}%)")
        print(f"   the two fields disagree on {vq_disagree} generator-years")
        if _q.notna().mean() * 100 > (_ans / _tot * 100 if _tot else 0):
            print("   The ratings carry the burden of evidence on capability;")
            print("   this field can corroborate but cannot establish it.")
        else:
            print("   The flag is answered at least as often as the rating is")
            print("   reported, so it is usable as independent evidence.")
    print("=" * 70)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
