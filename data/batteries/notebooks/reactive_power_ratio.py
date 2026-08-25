import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ==================================================================
    # reactive_power_ratio.py
    #
    # Reactive versus real power capability of the WECC battery fleet,
    # 2018-2022, and the per-node feasible-region parameters implied by it.
    #
    # PURPOSE
    #
    # A bulk nodal dispatch model represents each storage resource by a real
    # power variable and a reactive power variable confined to a feasible
    # region. The conventional region is the converter thermal limit
    #
    #     P^2 + Q^2 <= S^2
    #
    # a disc of radius S. Heating depends on total current and is indifferent
    # to the phase angle between voltage and current, which is what makes the
    # disc the right primitive. The disc reaches S on both axes, so adopting
    # it asserts that a unit's reactive rating equals its real power rating.
    #
    # This notebook tests that assertion and, where it fails, supplies the
    # correction. Per generator,
    #
    #     r = Q_nameplate / max(P_charge, P_discharge)
    #
    # the denominator being the binding real power limit of the region. Then
    #
    #     r ~ 1   the disc is correct as specified
    #     r < 1   reactive binds first; intersect the disc with |Q| <= Q_max
    #     r > 1   real power binds first; intersect with |P| <= P_max
    #
    # The disc is sized on max(P_max, Q_max) and the half planes placed at
    # min(P_max, Q_max). The region is therefore not recoverable from r alone;
    # both bounds are required and both are carried per node.
    #
    # UNITS
    #
    # EIA-860 Instructions, Schedule 3, line 38 defines the nameplate reactive
    # power rating in MVAR, not MVA. The distinction is material: under an MVA
    # reading S >= P holds identically and observations of Q > P would carry no
    # information, whereas under MVAR they are substantive statements about
    # converter capability.
    #

    #
    # INPUT
    #     battery_units_2018_2022.csv   generator-level unit table
    #     nodes.csv                     node coordinates, for the map only
    #
    # OUTPUT
    #     fleet_reactive_ratio_by_year.csv    fleet summary, five estimators
    #     node_reactive_bounds_by_year.csv    per-node model inputs
    #     reactive_screen_excluded.csv        records removed by the screen
    #     reactive_screen_sensitivity.csv     threshold sensitivity
    #     fig_ratio_by_year.png               estimators over time
    #     fig_ratio_distribution.png          per year and pooled
    #     fig_node_capacity_vs_ratio.png      per year and all years
    #     fig_node_spread_by_year.png         nodal dispersion
    #     fig_node_map.png                    per year and final year
    #     node_ratio_weighting_by_year.csv    weighted against unweighted
    # ==================================================================
    return


@app.cell
def _():
    # ===== imports, input, parameters =====
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

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

    units = pd.read_csv(PROC / "battery_units_2018_2022.csv")

    Q_COL = "Nameplate Reactive Power Rating"
    E_COL = "Nameplate Energy Capacity (MWh)"
    MW_COL = "Nameplate Capacity (MW)"
    CH_COL = "Maximum Charge Rate (MW)"
    DIS_COL = "Maximum Discharge Rate (MW)"

    # Converter reactive output is bounded by apparent power capacity, so a
    # ratio an order of magnitude above unity is not realisable and indicates a
    # units or decimal error. 
    SCREEN_RATIO = 10.0

    # Ratio at which the plotted axis is truncated. One node reports 6.67 and
    # compresses everything else into the bottom decile of the panel; clipped
    # points are drawn as triangles at the ceiling so none are hidden.
    PLOT_CEILING = 2.0

    # Node ratio carried into the maps. Both available columns are weighted and
    # they answer different questions:
    #   ratio_energy_weighted  per-generator ratios weighted by energy capacity
    #   ratio_aggregate        sum Q over sum max P, the quantity a nodal model
    #                          realises, identically the max_P-weighted mean
    # Unweighted per-node means are never mapped: they let a 0.3 MW unit count
    # as much as a 500 MW one, which misstates the system being modelled.
    MAP_RATIO = "ratio_energy_weighted"

    # State boundaries. Fetched once and cached beside the notebook; the layer
    # is only a backdrop and a spatial-join key, never an input to any figure.
    STATES_FILE = RAW / "us_states.geojson"
    STATES_URL = (
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
        "master/data/geojson/us-states.json"
    )

    print("unit table:", units.shape)
    print(
        "generator-years:",
        len(units),
        "| plants:",
        units["Plant Code"].nunique(),
        "| nodes:",
        units["geohash"].nunique(),
    )
    return (
        CH_COL,
        DIS_COL,
        E_COL,
        MAP_RATIO,
        MW_COL,
        PLOT_CEILING,
        PROC,
        Q_COL,
        RAW,
        SCREEN_RATIO,
        STATES_FILE,
        STATES_URL,
        TwoSlopeNorm,
        pd,
        plt,
        units,
    )


@app.cell
def _(pd, units):
    # ===== symmetry of the reactive bound =====
    # The region is symmetric about the real power axis only if the source
    # reports a single reactive rating. Separate leading and lagging ratings
    # would place the half planes at different heights. Numeric ratings are
    # separated from categorical columns by dtype because both carry the word
    # "reactive" and only the former bounds the region.
    _mentions = [
        c for c in units.columns if "reactive" in c.lower() or "mvar" in c.lower()
    ]
    _ratings = [c for c in _mentions if pd.api.types.is_numeric_dtype(units[c])]
    _flags = [c for c in _mentions if c not in _ratings]

    print("columns mentioning reactive quantities:", len(_mentions))
    print("  numeric ratings :", _ratings if _ratings else "none")
    print("  categorical     :", _flags if _flags else "none")
    if len(_ratings) == 1:
        print(
            "\nA single reported rating. The region is modelled as symmetric"
            "\nabout the real power axis, bounded by +/- the same Q_max."
        )
    else:
        print(
            "\nMore than one numeric rating. If these are leading and lagging"
            "\ncapability the region is asymmetric in Q and the half planes"
            "\nmust be placed independently."
        )
    return


@app.cell
def _(CH_COL, DIS_COL, E_COL, Q_COL, SCREEN_RATIO, units):
    # ===== per-generator ratio and record classification =====
    #
    # The denominator is max(charge, discharge): the two directional ratings
    # are usually equal, and where they differ the larger is the binding real
    # power limit and therefore the correct scale for the region.
    #
    # A reported zero declares no reactive capability and is retained with
    # ratio zero. Only absent values are excluded.
    ratios = units.copy()
    ratios["max_P_MW"] = ratios[[CH_COL, DIS_COL]].max(axis=1)
    ratios["ratio_Q_over_P"] = ratios[Q_COL] / ratios["max_P_MW"]

    _defined = (
        ratios[Q_COL].notna()
        & ratios["max_P_MW"].notna()
        & ratios["max_P_MW"].gt(0)
    )
    ratios["ratio_defined"] = _defined
    ratios["implausible"] = _defined & ratios["ratio_Q_over_P"].gt(SCREEN_RATIO)
    ratios["usable"] = _defined & ~ratios["implausible"]
    # Energy weighting is applied to the retained sample, so weight_ok is
    # conditioned on usable rather than on ratio_defined.
    ratios["weight_ok"] = (
        ratios["usable"] & ratios[E_COL].notna() & ratios[E_COL].gt(0)
    )

    print("generator-years:", len(ratios))
    print(f"  reactive rating absent         : {int(ratios[Q_COL].isna().sum())}")
    print(f"  reactive rating declared zero  : {int((ratios[Q_COL] == 0).sum())}")
    print(f"  real power rating absent or 0  : {int((~ratios['max_P_MW'].gt(0)).sum())}")
    print(f"  energy capacity absent or 0    : {int((~ratios[E_COL].gt(0)).sum())}")
    print(f"\n  ratio defined                  : {int(_defined.sum())}")
    print(f"  retained after screening       : {int(ratios['usable'].sum())}")
    print(f"  admissible for energy weighting: {int(ratios['weight_ok'].sum())}")

    print(
        "\ndirectional ratings differ on",
        int((ratios[CH_COL] != ratios[DIS_COL]).sum()),
        "generator-years",
    )
    print(
        "reactive rating exceeds real power rating on",
        int((ratios[Q_COL] > ratios["max_P_MW"]).sum()),
        "of",
        int(_defined.sum()),
        "generator-years reporting both",
    )
    return (ratios,)


@app.cell
def _(CH_COL, DIS_COL, E_COL, Q_COL, SCREEN_RATIO, ratios):
    # ===== records removed by the screen =====
    # Each excluded record is enumerated with the fields motivating exclusion
    # and its share of the annual reactive total. Large reactive contribution
    # against negligible real capacity is the signature of a units or decimal
    # error. If a record is later confirmed as hardware, raise SCREEN_RATIO.
    excluded = ratios[ratios["implausible"]].copy()

    print(f"screening rule: ratio_Q_over_P > {SCREEN_RATIO}")
    print("generator-years excluded:", len(excluded))

    if len(excluded):
        print()
        print(
            excluded[
                [
                    "Year",
                    "Plant Name",
                    "State",
                    "geohash",
                    CH_COL,
                    DIS_COL,
                    Q_COL,
                    E_COL,
                    "ratio_Q_over_P",
                ]
            ]
            .sort_values("ratio_Q_over_P", ascending=False)
            .to_string(index=False)
        )

        print("\ncontribution of the excluded records")
        for _y, _g in ratios[ratios["ratio_defined"]].groupby("Year"):
            _b = _g[_g["implausible"]]
            if len(_b):
                print(
                    f"  {_y}: {_b[Q_COL].sum():.1f} of {_g[Q_COL].sum():.1f} MVAR"
                    f" ({_b[Q_COL].sum() / _g[Q_COL].sum() * 100:.1f}%) from"
                    f" {len(_b)} generator-year(s) holding"
                    f" {_b['max_P_MW'].sum():.1f} MW"
                )
    else:
        print("no records exceed the threshold.")
    return


@app.cell
def _(E_COL, MW_COL, Q_COL, pd):
    # ===== estimator definitions =====
    #
    # Five summaries of one quantity. They are not interchangeable and their
    # disagreement is diagnostic:
    #
    #   energy_weighted  weighted by nameplate energy capacity. Storage is an
    #                    energy-limited device, so this weights by the quantity
    #                    being represented
    #   power_weighted   weighted by nameplate real power capacity
    #   unweighted_mean  one weight per generator; dominated by small units
    #   median           robust, but pinned by the mass at exactly unity
    #   aggregate        sum of reactive ratings over sum of real power ratings
    #
    # The aggregate is what a bulk nodal model realises: one node carries one
    # reactive budget and one power budget and the model forms their quotient.
    # It is identically the max_P-weighted mean of the per-generator ratios,
    # which is distinct from the nameplate-MW-weighted mean; both are reported.
    def weighted_mean(frame, value, weight):
        _ok = frame[value].notna() & frame[weight].notna() & frame[weight].gt(0)
        if not _ok.any():
            return float("nan")
        return (frame.loc[_ok, value] * frame.loc[_ok, weight]).sum() / frame.loc[
            _ok, weight
        ].sum()

    def summarise(frame):
        _d = frame[frame["usable"]]
        return pd.Series(
            {
                "generators": len(_d),
                "energy_weighted": weighted_mean(_d, "ratio_Q_over_P", E_COL),
                "power_weighted": weighted_mean(_d, "ratio_Q_over_P", MW_COL),
                "unweighted_mean": _d["ratio_Q_over_P"].mean(),
                "median": _d["ratio_Q_over_P"].median(),
                "aggregate_sumQ_over_sumP": _d[Q_COL].sum() / _d["max_P_MW"].sum(),
            }
        )

    def summarise_unscreened(frame):
        _d = frame[frame["ratio_defined"]]
        return pd.Series(
            {
                "generators": len(_d),
                "energy_weighted": weighted_mean(_d, "ratio_Q_over_P", E_COL),
                "power_weighted": weighted_mean(_d, "ratio_Q_over_P", MW_COL),
                "unweighted_mean": _d["ratio_Q_over_P"].mean(),
                "median": _d["ratio_Q_over_P"].median(),
                "aggregate_sumQ_over_sumP": _d[Q_COL].sum() / _d["max_P_MW"].sum(),
            }
        )

    return summarise, summarise_unscreened, weighted_mean


@app.cell
def _(ratios, summarise, summarise_unscreened):
    # ===== fleet summary by year =====
    # Screened and unscreened results are shown together so the influence of
    # the excluded records is visible without recomputation.
    fleet_ratio = (
        ratios.groupby("Year").apply(summarise, include_groups=False).round(4)
    )
    fleet_ratio["generators"] = fleet_ratio["generators"].astype(int)
    fleet_ratio["screened_out"] = (
        ratios[ratios["implausible"]]
        .groupby("Year")
        .size()
        .reindex(fleet_ratio.index, fill_value=0)
        .astype(int)
    )

    fleet_unscreened = (
        ratios.groupby("Year")
        .apply(summarise_unscreened, include_groups=False)
        .round(4)
    )

    print("WECC battery fleet, reactive rating over max real power rating")
    print("\nscreened:")
    print(fleet_ratio.to_string())
    print("\nunscreened, for comparison:")
    print(fleet_unscreened.to_string())

    print("\nall years pooled, screened:")
    print(summarise(ratios).round(4).to_string())
    return (fleet_ratio,)


@app.cell
def _(fleet_ratio, plt):
    # ===== figure 1: estimators over time =====
    # One panel; the screened estimators are the reported result and the
    # unscreened series are omitted rather than shown alongside.
    fig_trend, _ax = plt.subplots(figsize=(8.5, 4.6))

    for _c, _m in [
        ("energy_weighted", "o"),
        ("aggregate_sumQ_over_sumP", "s"),
        ("power_weighted", "^"),
        ("unweighted_mean", "v"),
        ("median", "d"),
    ]:
        _ax.plot(fleet_ratio.index, fleet_ratio[_c], marker=_m, lw=1.6, ms=5, label=_c)

    _ax.axhline(1.0, color="0.35", ls="--", lw=1)
    _ax.set_xticks(list(fleet_ratio.index))
    _ax.set_ylabel("Q / max(P)")
    _ax.legend(fontsize=8, frameon=False)
    _ax.grid(alpha=0.25)
    _ax.set_title(
        "WECC battery fleet: reactive rating over max real power rating, by year, screened",
        fontsize=11,
    )
    fig_trend.tight_layout()
    fig_trend
    return (fig_trend,)


@app.cell
def _(E_COL, Q_COL, pd, ratios, weighted_mean):
    # ===== sensitivity to the screening threshold =====
    # The threshold is a modelling choice and is treated as one. Stability
    # across an order of magnitude of candidate values means the specific
    # number is not load bearing. The unweighted mean is included because it is
    # the estimator most exposed to the excluded records, and the contrast with
    # the weighted columns is itself informative.
    _rows = []
    for _t in [float("inf"), 50.0, 20.0, 10.0, 5.0, 3.0, 2.0]:
        _d = ratios[ratios["ratio_defined"] & ratios["ratio_Q_over_P"].le(_t)]
        _rows.append(
            {
                "threshold": _t,
                "generators": len(_d),
                "dropped": int(ratios["ratio_defined"].sum()) - len(_d),
                "energy_weighted": weighted_mean(_d, "ratio_Q_over_P", E_COL),
                "aggregate": _d[Q_COL].sum() / _d["max_P_MW"].sum(),
                "unweighted_mean": _d["ratio_Q_over_P"].mean(),
            }
        )
    screen_sensitivity = pd.DataFrame(_rows).round(4)
    print("sensitivity of pooled estimates to the screening threshold")
    print(screen_sensitivity.to_string(index=False))
    return (screen_sensitivity,)


@app.cell
def _(pd, ratios):
    # ===== distribution of the ratio =====
    # A central estimate does not determine the region: a fleet reporting unity
    # throughout and a fleet split between zero and two share a mean of one and
    # require different treatments.
    #
    # The band edges at 0.999 and 1.001 isolate records whose reactive rating
    # equals the real power rating exactly; these are counted apart from
    # records merely close to it.
    _d = ratios[ratios["usable"]]

    print("distribution of Q / max(P), screened generator-years")
    print(
        _d["ratio_Q_over_P"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(3)
        .to_string()
    )

    _bins = [-0.001, 0.001, 0.5, 0.9, 0.999, 1.001, 1.1, 2.0, 5.0, float("inf")]
    _labels = [
        "exactly 0",
        "0 to 0.5",
        "0.5 to 0.9",
        "0.9 to 1.0",
        "exactly 1.0",
        "1.0 to 1.1",
        "1.1 to 2",
        "2 to 5",
        "above 5",
    ]
    _band = _d.groupby("Year")["ratio_Q_over_P"].apply(
        lambda s: s.groupby(pd.cut(s, _bins, labels=_labels), observed=False).size()
    )
    print("\ngenerator-years by ratio band")
    print(_band.unstack(fill_value=0).to_string())

    _near = int(_d["ratio_Q_over_P"].between(0.9, 1.1).sum())
    _exact = int((_d["ratio_Q_over_P"] == 1.0).sum())
    print(f"\nwithin 10 percent of unity : {_near} / {len(_d)}"
          f" ({_near / len(_d) * 100:.1f}%)")
    print(f"reporting exactly 1.000    : {_exact} / {len(_d)}"
          f" ({_exact / len(_d) * 100:.1f}%)")
    return


@app.cell
def _(plt, ratios):
    # ===== figure 2: distribution, per year and pooled =====
    # One panel per year on a shared axis, plus the pooled panel. A common
    # x-axis is what makes the migration of mass out of unity visible.
    _d = ratios[ratios["usable"]]
    _years = sorted(_d["Year"].unique())
    _ncol = 3
    _nrow = -(-(len(_years) + 1) // _ncol)

    fig_dist, _axes = plt.subplots(
        _nrow, _ncol, figsize=(12, 3.1 * _nrow), sharex=True
    )
    _flat = _axes.ravel()

    for _i, _y in enumerate(_years):
        _s = _d.loc[_d["Year"] == _y, "ratio_Q_over_P"].clip(upper=3.0)
        _ax = _flat[_i]
        _ax.hist(_s, bins=48, range=(0, 3), color="#4a6f8a",
                 edgecolor="white", linewidth=0.3)
        _ax.axvline(1.0, color="#b4442e", ls="--", lw=1.1)
        _exact = int((_d.loc[_d["Year"] == _y, "ratio_Q_over_P"] == 1.0).sum())
        _ax.set_title(f"{_y}   n = {len(_s)}, exactly 1.000: {_exact}", fontsize=9)
        _ax.grid(alpha=0.18)

    _ax = _flat[len(_years)]
    _pool = _d["ratio_Q_over_P"].clip(upper=3.0)
    _ax.hist(_pool, bins=48, range=(0, 3), color="#3c6e88",
             edgecolor="white", linewidth=0.3)
    _ax.axvline(1.0, color="#b4442e", ls="--", lw=1.1)
    _ax.set_title(
        f"all years   n = {len(_pool)},"
        f" exactly 1.000: {int((_d['ratio_Q_over_P'] == 1.0).sum())}",
        fontsize=9,
    )
    _ax.grid(alpha=0.18)

    for _j in range(len(_years) + 1, len(_flat)):
        _flat[_j].axis("off")
    for _ax in _flat[: len(_years) + 1]:
        _ax.set_xlabel("Q / max(P)", fontsize=8)
        _ax.tick_params(labelsize=8)
    fig_dist.suptitle(
        "Distribution of Q / max(P); values above 3 plotted at 3", fontsize=11
    )
    fig_dist.tight_layout()
    fig_dist
    return (fig_dist,)


@app.cell
def _(CH_COL, DIS_COL, E_COL, MW_COL, Q_COL, ratios, weighted_mean):
    # ===== node-level bounds =====
    #
    # Aggregation is by summation of the directional ratings. A node charges
    # with all co-located units charging simultaneously, so the nodal charging
    # limit is the sum of unit charging rates, and likewise for discharge.
    # Taking max(charge, discharge) per unit before summation would combine two
    # mutually exclusive operating states and yield a bound corresponding to no
    # realisable condition.
    #
    # Charging and discharging bounds are separate columns because they are not
    # in general equal; the box constraint on the real power variable is
    # correspondingly asymmetric.
    _d = ratios[ratios["usable"]]

    node_bounds = (
        _d.groupby(["geohash", "Year"])
        .agg(
            generators=("Plant Code", "size"),
            nameplate_MW=(MW_COL, "sum"),
            energy_MWh=(E_COL, "sum"),
            max_charge_MW=(CH_COL, "sum"),
            max_discharge_MW=(DIS_COL, "sum"),
            reactive_MVAR=(Q_COL, "sum"),
        )
        .reset_index()
    )
    node_bounds["max_P_MW"] = node_bounds[["max_charge_MW", "max_discharge_MW"]].max(
        axis=1
    )
    node_bounds["ratio_aggregate"] = (
        node_bounds["reactive_MVAR"] / node_bounds["max_P_MW"]
    )

    _ew = (
        _d.groupby(["geohash", "Year"])
        .apply(
            lambda g: weighted_mean(g, "ratio_Q_over_P", E_COL), include_groups=False
        )
        .rename("ratio_energy_weighted")
        .reset_index()
    )
    node_bounds = node_bounds.merge(_ew, on=["geohash", "Year"], how="left")

    _drop = (
        ratios[ratios["implausible"]]
        .groupby(["geohash", "Year"])
        .size()
        .rename("screened_out")
        .reset_index()
    )
    node_bounds = node_bounds.merge(_drop, on=["geohash", "Year"], how="left")
    node_bounds["screened_out"] = node_bounds["screened_out"].fillna(0).astype(int)

    node_bounds = node_bounds.round(
        {
            "nameplate_MW": 2,
            "energy_MWh": 2,
            "max_charge_MW": 2,
            "max_discharge_MW": 2,
            "reactive_MVAR": 2,
            "max_P_MW": 2,
            "ratio_aggregate": 4,
            "ratio_energy_weighted": 4,
        }
    )

    print("node bounds table:", node_bounds.shape,
          "| nodes:", node_bounds["geohash"].nunique())
    print("\nnode-years with asymmetric real power bounds:",
          int((node_bounds["max_charge_MW"] != node_bounds["max_discharge_MW"]).sum()))
    print("node-years with zero reactive capability:",
          int((node_bounds["reactive_MVAR"] == 0).sum()))
    print("  the region there is a segment of the real power axis;"
          " ratio_aggregate is zero, not missing")

    # A node-year whose every record is screened leaves node_bounds entirely,
    # so screened_out cannot record it. Those cases are listed here; without
    # this the audit trail has a gap exactly where the screen bit hardest.
    _all_ny = set(
        map(tuple, ratios.loc[ratios["ratio_defined"], ["geohash", "Year"]].values)
    )
    _kept_ny = set(map(tuple, node_bounds[["geohash", "Year"]].values))
    _lost = sorted(_all_ny - _kept_ny)
    print("\nnode-years eliminated entirely by the screen:", len(_lost))
    for _g, _y in _lost:
        print(f"  {_g}  {_y}")
    node_bounds.head(12)
    return (node_bounds,)


@app.cell
def _(node_bounds):
    # ===== heterogeneity across nodes =====
    # Two questions, answered separately: how a node drawn at random behaves,
    # and how the installed capacity behaves. A dispatch model is exposed to
    # the second. Where they diverge, the capacity-weighted figure governs.
    print("node-level aggregate ratio, distribution by year")
    print(
        node_bounds.groupby("Year")["ratio_aggregate"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .round(3)
        .to_string()
    )

    _rows = []
    for _y, _g in node_bounds.groupby("Year"):
        _w = _g["nameplate_MW"]
        _big = _g.nlargest(10, "nameplate_MW")
        _rows.append(
            {
                "Year": _y,
                "nodes": len(_g),
                "unweighted": _g["ratio_aggregate"].mean(),
                "capacity_weighted": (_g["ratio_aggregate"] * _w).sum() / _w.sum(),
                "top10_share_MW": _big["nameplate_MW"].sum() / _w.sum(),
                "top10_weighted": (_big["ratio_aggregate"] * _big["nameplate_MW"]).sum()
                / _big["nameplate_MW"].sum(),
                "below_0.9": int(_g["ratio_aggregate"].lt(0.9).sum()),
                "within_10pct": int(_g["ratio_aggregate"].between(0.9, 1.1).sum()),
                "above_1.1": int(_g["ratio_aggregate"].gt(1.1).sum()),
            }
        )
    import pandas as _pd

    node_year_summary = _pd.DataFrame(_rows).set_index("Year").round(4)
    print("\nunweighted against capacity-weighted, by year")
    print(node_year_summary.to_string())
    print(
        "\nA capacity-weighted mean below the unweighted mean means low-ratio"
        "\nnodes carry the capacity."
    )

    _last_year = int(node_bounds["Year"].max())
    _last = node_bounds[node_bounds["Year"] == _last_year]
    print(f"\nlargest nodes by installed capacity, {_last_year}")
    print(
        _last.nlargest(10, "nameplate_MW")[
            ["geohash", "generators", "nameplate_MW", "max_P_MW", "reactive_MVAR",
             "ratio_aggregate", "ratio_energy_weighted"]
        ].to_string(index=False)
    )
    return (node_year_summary,)


@app.cell
def _(RAW, node_bounds, pd):
    # ===== node coordinates =====
    # Coordinates are used for the map only and never for aggregation, so a
    # failure here must not silently propagate into the numeric results. The
    # join is reported and the map cell degrades if it is incomplete.
    try:
        _nodes = pd.read_csv(RAW / "nodes.csv")
        _cols = {c.lower(): c for c in _nodes.columns}
        _key = _cols.get("geocode") or _cols.get("geohash")
        _lat = _cols.get("lat") or _cols.get("latitude")
        _lon = _cols.get("long") or _cols.get("lon") or _cols.get("longitude")
        node_coords = (
            _nodes[[_key, _lat, _lon]]
            .rename(columns={_key: "geohash", _lat: "Lat", _lon: "Long"})
            .drop_duplicates("geohash")
        )
        _have = set(node_coords["geohash"])
        _want = set(node_bounds["geohash"])
        print("nodes.csv:", _nodes.shape, "| usable coordinates:", len(node_coords))
        print("nodes in bounds table:", len(_want),
              "| matched:", len(_want & _have),
              "| unmatched:", len(_want - _have))
        if _want - _have:
            print("  unmatched:", sorted(_want - _have))
    except FileNotFoundError:
        node_coords = pd.DataFrame(columns=["geohash", "Lat", "Long"])
        print("nodes.csv not found; the map cell will be skipped.")
    return (node_coords,)


@app.cell
def _(PLOT_CEILING, node_bounds, plt):
    # ===== figure 3: node ratio against node capacity, per year and pooled =====
    # The question is whether the disc fails where the capacity is. Nodes above
    # PLOT_CEILING are drawn as triangles on the ceiling: without truncation a
    # single node near 6.7 compresses the entire informative range.
    _years = sorted(node_bounds["Year"].unique())
    _ncol = 3
    _nrow = -(-(len(_years) + 1) // _ncol)

    fig_nodes, _axes = plt.subplots(
        _nrow, _ncol, figsize=(12, 3.4 * _nrow), sharex=True, sharey=True
    )
    _flat = _axes.ravel()

    def _panel(ax, frame, title):
        _lo = frame[frame["ratio_aggregate"] <= PLOT_CEILING]
        _hi = frame[frame["ratio_aggregate"] > PLOT_CEILING]
        ax.scatter(
            _lo["nameplate_MW"], _lo["ratio_aggregate"],
            s=18 + 7 * _lo["generators"], alpha=0.7, color="#3c6e88",
            edgecolor="white", linewidth=0.5,
        )
        if len(_hi):
            ax.scatter(
                _hi["nameplate_MW"], [PLOT_CEILING] * len(_hi),
                s=34, marker="^", color="#b4442e", alpha=0.85,
            )
        ax.axhline(1.0, color="#b4442e", ls="--", lw=1.1)
        _w = frame["nameplate_MW"]
        ax.axhline(
            (frame["ratio_aggregate"] * _w).sum() / _w.sum(),
            color="#1d7a5f", ls="-", lw=1.2,
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.08, PLOT_CEILING + 0.12)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)

    for _i, _y in enumerate(_years):
        _g = node_bounds[node_bounds["Year"] == _y]
        _panel(_flat[_i], _g, f"{_y}   {len(_g)} nodes")

    _panel(_flat[len(_years)], node_bounds,
           f"all years   {len(node_bounds)} node-years")

    for _j in range(len(_years) + 1, len(_flat)):
        _flat[_j].axis("off")
    for _ax in _flat[: len(_years) + 1]:
        _ax.set_xlabel("node capacity, MW (log)", fontsize=8)
        _ax.tick_params(labelsize=8)
    _flat[0].set_ylabel("sum Q / sum max P", fontsize=8)

    fig_nodes.suptitle(
        "Node ratio against node size. Dashed: unity. Solid: capacity-weighted"
        f" mean. Triangles: ratio above {PLOT_CEILING}",
        fontsize=10,
    )
    fig_nodes.tight_layout()
    fig_nodes
    return (fig_nodes,)


@app.cell
def _(STATES_FILE, STATES_URL):
    # ===== state boundary layer =====
    # Real boundaries from a published GeoJSON rather than reconstructed ones.
    # Fetched once and cached beside the notebook. The layer is a backdrop
    # only: no figure value and no table depends on it, so a failure here costs
    # the outlines and nothing else.
    import os
    import urllib.request

    try:
        import geopandas as gpd

        if not os.path.exists(STATES_FILE):
            print("fetching state boundaries ...")
            with urllib.request.urlopen(STATES_URL, timeout=60) as _r:
                with open(STATES_FILE, "wb") as _f:
                    _f.write(_r.read())
        states = gpd.read_file(STATES_FILE)
        print("state layer:", len(states), "polygons from", STATES_FILE)
    except Exception as _e:
        states = None
        print("state boundaries unavailable:", type(_e).__name__, _e)
        print(f"  install geopandas, or download {STATES_URL}")
        print(f"  and save it beside this notebook as {STATES_FILE}")
        print("  the node map will still be drawn, without outlines")
    return (states,)


@app.cell
def _(
    MAP_RATIO,
    PLOT_CEILING,
    TwoSlopeNorm,
    node_bounds,
    node_coords,
    plt,
    states,
):
    # ===== figure 5: nodes on the map, per year and final year =====
    #
    # Colour is the weighted node ratio on a diverging scale centred at unity:
    # blue needs the horizontal cut, red the vertical one. Marker area is
    # installed capacity, so a large blue marker marks a node where the disc
    # overstates reactive headroom behind a lot of storage. Reading colour and
    # size together is the point of the figure; colour alone would give a small
    # experimental installation the same visual weight as a 1.2 GW node.
    _WEST = dict(xlim=(-126, -101), ylim=(30.5, 49.8))

    def _base(ax):
        if states is not None:
            states.boundary.plot(ax=ax, color="0.72", linewidth=0.5, zorder=1)
        ax.set_xlim(*_WEST["xlim"])
        ax.set_ylim(*_WEST["ylim"])
        ax.set_aspect(1 / __import__("numpy").cos(__import__("numpy").deg2rad(40.0)))
        ax.set_xticks([-124, -118, -112, -106])
        ax.set_yticks([32, 36, 40, 44, 48])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.12, lw=0.5)

    if len(node_coords) == 0:
        fig_map = None
        print("no coordinates available; node map skipped.")
    else:
        _m = node_bounds.merge(node_coords, on="geohash", how="inner")
        _years = sorted(_m["Year"].unique())
        _norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=PLOT_CEILING)
        _smax = _m["nameplate_MW"].max()
        _ncol = 3
        _nrow = -(-(len(_years) + 1) // _ncol)

        fig_map, _axes = plt.subplots(_nrow, _ncol, figsize=(12, 4.6 * _nrow))
        _flat = _axes.ravel()

        def _draw(ax, frame):
            return ax.scatter(
                frame["Long"], frame["Lat"],
                s=14 + 150 * (frame["nameplate_MW"] / _smax) ** 0.5,
                c=frame[MAP_RATIO].clip(upper=PLOT_CEILING),
                cmap="RdYlBu_r", norm=_norm,
                edgecolor="0.25", linewidth=0.4, alpha=0.9, zorder=3,
            )

        for _i, _y in enumerate(_years):
            _g = _m[_m["Year"] == _y]
            _base(_flat[_i])
            _sc = _draw(_flat[_i], _g)
            _w = _g["nameplate_MW"]
            _flat[_i].set_title(
                f"{_y}   {len(_g)} nodes, {_w.sum():,.0f} MW\n"
                f"capacity-weighted ratio {(_g[MAP_RATIO] * _w).sum() / _w.sum():.3f}",
                fontsize=9,
            )

        _last_year = int(_m["Year"].max())
        _g = _m[_m["Year"] == _last_year]
        _ax = _flat[len(_years)]
        _base(_ax)
        _draw(_ax, _g)
        for _, _r in _g.nlargest(8, "nameplate_MW").iterrows():
            _ax.annotate(
                f"{_r['geohash']}\n{_r['nameplate_MW']:,.0f} MW,"
                f" {_r[MAP_RATIO]:.2f}",
                (_r["Long"], _r["Lat"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=6, color="0.2",
            )
        _ax.set_title(f"{_last_year}, eight largest nodes labelled", fontsize=9)

        for _j in range(len(_years) + 1, len(_flat)):
            _flat[_j].axis("off")

        _cb = fig_map.colorbar(
            _sc, ax=_axes, orientation="horizontal",
            fraction=0.035, pad=0.05, aspect=45,
        )
        _cb.set_label(
            f"{MAP_RATIO}.  Blue: reactive binds, cut horizontally."
            "  Red: real power binds, cut vertically.",
            fontsize=8,
        )
        _cb.ax.tick_params(labelsize=7)
        fig_map.suptitle(
            "WECC storage nodes: weighted reactive capability relative to real"
            " power. Marker area is installed capacity.",
            fontsize=11,
        )
    fig_map
    return (fig_map,)


@app.cell
def _(
    PROC,
    fig_dist,
    fig_map,
    fig_nodes,
    fig_trend,
    fleet_ratio,
    node_bounds,
    node_year_summary,
    screen_sensitivity,
):
    # ===== outputs =====
    fleet_ratio.reset_index().to_csv(
            PROC / "fleet_reactive_ratio_by_year.csv", index=False
        )
    node_bounds.to_csv(PROC / "node_reactive_bounds_by_year.csv", index=False)
    node_bounds.to_csv(PROC / "node_reactive_bounds_by_year.csv", index=False)
    screen_sensitivity.to_csv(PROC / "reactive_screen_sensitivity.csv", index=False)
    node_year_summary.reset_index().to_csv(
            PROC / "node_ratio_weighting_by_year.csv", index=False
        )

    fig_trend.savefig(PROC / "fig_ratio_by_year.png", dpi=150, bbox_inches="tight")
    fig_dist.savefig(PROC / "fig_ratio_distribution.png", dpi=150, bbox_inches="tight")
    fig_nodes.savefig(
            PROC / "fig_node_capacity_vs_ratio.png", dpi=150, bbox_inches="tight"
        )
    if fig_map is not None:
        fig_map.savefig(PROC / "fig_node_map.png", dpi=150, bbox_inches="tight")

    print("written")
    print(f"  fleet_reactive_ratio_by_year.csv    {len(fleet_ratio):5d} rows")
    print(f"  node_reactive_bounds_by_year.csv    {len(node_bounds):5d} rows")
    print(f"  node_ratio_weighting_by_year.csv    {len(node_year_summary):5d} rows")
    print(f"  reactive_screen_sensitivity.csv     {len(screen_sensitivity):5d} rows")
    print(f"  figures                             {4 + (fig_map is not None)}")
    print()
    print("node_reactive_bounds_by_year.csv supplies the model inputs:")
    print("  max_charge_MW, max_discharge_MW  box bounds on the real power variable")
    print("  reactive_MVAR                    bound on the reactive power variable")
    print("  max_P_MW                         with reactive_MVAR, sizes the disc")
    print("  ratio_aggregate                  orientation of the half-plane cut")
    print("  screened_out                     records removed by the filing screen")
    return


if __name__ == "__main__":
    app.run()
