import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # ==================================================================
    # tech_and_applications.py
    #
    # Storage technology mix and declared applications for the WECC battery
    # fleet, 2018-2022.
    #
    # PURPOSE
    #
    # Two parameters of the nodal dispatch model are settled here.
    #
    # 1. Round-trip efficiency. The state-of-charge recursion carries three
    #    efficiency terms,
    #
    #        q_{t+1} = eta_s q_t + delta ( eta_c c_t - (1/eta_d) d_t )
    #
    #    a self-discharge term and a charge and discharge pair. A bulk model
    #    represents the fleet with one blended value rather than a per-unit
    #    one, and the blend depends on what fraction of the fleet is each
    #    chemistry. This notebook supplies that fraction.
    #
    # 2. Four-quadrant operation. The reactive support application flag is the
    #    only field in which an operator states whether the installation is
    #    intended to provide voltage support at all. It is therefore the one
    #    independent check available on the assumption that storage can operate
    #    in all four quadrants of the P-Q plane, which the companion notebook
    #    reactive_power_ratio.py examines from the ratings instead.
    #
    # WEIGHTING
    #
    # The mix is reported under three denominators: generator count, nameplate
    # power, and nameplate energy. They disagree substantially, and only the
    # last two describe the system. Efficiency acts on energy throughput, so
    # the energy share is the correct weight for the blend; the generator-count
    # share is reported solely to show how far it misleads. A fleet that is a
    # fifth non-lithium by unit count can be a two-hundredth non-lithium by
    # energy, and the parameter follows the second figure.
    #
    # RESPONSE RATES
    #
    # The application fields are voluntary and mostly blank. A yes-share
    # computed over respondents alone is a statement about a self-selected
    # minority, not about the fleet, so every proportion below is reported
    # beside the response rate that produced it. Rows are ordered by response
    # rate for the same reason.
    #
    # REVISION. Seven changes after reviewing the first run.
    #
    #  1. Coverage is now reported in energy as well as in units. A blend
    #     computed over the generator-years that declare a chemistry is a
    #     statement about that subset; how large the subset is, measured in the
    #     quantity being weighted, is what says whether it stands for the fleet.
    #
    #  2. The efficiency blend is differenced before rounding. The headline
    #     adjustment is a fraction of an efficiency point, which is finer than
    #     the resolution the previous ordering left it.
    #
    #  3. The pessimistic perturbation value is a named parameter rather than a
    #     literal inside the cell that uses it.
    #
    #  4. Co-declared technologies are counted distinctly. A generator naming
    #     the same chemistry in two columns is single-technology, and counting
    #     non-null cells reported it as two.
    #
    #  5. Node-level chemical heterogeneity is carried out of its cell and into
    #     the findings, weighted by capacity. Whether a single efficiency per
    #     node is defensible is a question about where the capacity sits, not
    #     about how many nodes are mixed.
    #
    #  6. The application flags are normalised once into a long frame from
    #     which every downstream table is derived. There were three
    #     implementations of the same parse, and values that are neither Y, N
    #     nor blank were being counted as blank by all of them.
    #
    #  7. Ratios that can meet an empty denominator are guarded, and column
    #     names used in more than one cell are parameters rather than literals.
    #
    # INPUT
    #     battery_units_2018_2022.csv        generator-level unit table
    #
    # OUTPUT
    #     technology_mix_by_year.csv         mix under three denominators
    #     technology_efficiency_blend.csv    implied blended efficiency
    #     technology_node_mixing.csv         chemically mixed nodes
    #     applications_summary.csv           flags with response rates
    #     applications_by_year.csv           flags by year
    #     fig_technology_mix.png             count against capacity, by year
    #     fig_lithium_share.png              share under each denominator
    #     fig_application_response.png       response rate against yes-share
    #     fig_reactive_application.png       reactive support, per year
    #
    # Implementation note: marimo notebook; each name is bound in exactly one
    # cell and cell-local names are underscore prefixed.
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

    MW_COL = "Nameplate Capacity (MW)"
    E_COL = "Nameplate Energy Capacity (MWh)"
    Q_COL = "Nameplate Reactive Power Rating"

    # EIA-860 Instructions, Schedule 3, Table 5b.
    TECH_NAMES = {
        "LIB": "lithium ion",
        "FLB": "flow",
        "NAS": "sodium sulfur",
        "PBA": "lead acid",
        "NIC": "nickel based",
        "ZIB": "zinc",
        "MAB": "metal air",
        "OTH": "other",
    }
    ENCLOSURE_NAMES = {
        "BL": "building",
        "CT": "containerised",
        "OT": "other",
    }

    # Placeholder AC-to-AC round-trip efficiencies used to convert the mix into
    # a single blended parameter. THESE ARE NOT DATA. They are order-of-
    # magnitude values standing in for a cited source, and the blend below is
    # reported as conditional on them. Substitute project values before use;
    # the cell that consumes them also reports how much the blend moves under
    # a wide perturbation, which is the figure that determines whether the
    # choice matters at all.
    ROUND_TRIP = {
        "lithium ion": 0.86,
        "flow": 0.70,
        "sodium sulfur": 0.80,
        "lead acid": 0.78,
        "nickel based": 0.70,
        "zinc": 0.70,
        "metal air": 0.60,
        "other": 0.75,
    }

    REFERENCE_TECH = "lithium ion"

    # Efficiency assigned to every non-reference chemistry in the perturbation
    # column. Chosen below any plausible value so the column is an upper bound
    # on the error from the table above rather than an alternative estimate.
    PESSIMISTIC_RTE = 0.50

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
        ENCLOSURE_NAMES,
        E_COL,
        MW_COL,
        PESSIMISTIC_RTE,
        PROC,
        Q_COL,
        REFERENCE_TECH,
        ROUND_TRIP,
        TECH_NAMES,
        pd,
        plt,
        units,
    )


@app.cell
def _(units):
    # ===== column discovery =====
    # EIA column names have moved between vintages, so the fields are located
    # by pattern and the result is printed. A silent mismatch here would
    # produce empty tables rather than an error, which is the failure mode this
    # cell exists to prevent.
    #
    # The technology columns are ordered by their trailing index rather than by
    # their position in the file, because the first of them is taken as the
    # primary declaration and file order is not guaranteed to preserve that.
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
        "Ramping/Spinning Reserve",
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
        # Fall back on shape: a flag column contains only Y, N or blank. Fields
        # already identified as something else are excluded, since a technology
        # or enclosure column could satisfy that test by accident.
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
    #
    # The form allows up to four technologies per generator. The first field is
    # taken as the primary declaration and carries the mix; the count of
    # generators declaring more than one is reported separately, since a unit
    # split across chemistries would need an apportionment rule and the mix
    # would no longer be a partition.
    #
    # Multiplicity counts distinct codes, not populated cells: the same
    # chemistry named in two columns is one chemistry, and counting cells
    # reports a homogeneous unit as mixed.
    #
    # Coverage is reported in energy as well as in units. Everything downstream
    # is computed over the generator-years that declare a chemistry, so the
    # share of fleet energy those records carry is what says whether the result
    # stands for the fleet or for a subset of it.
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
    # nunique over the row counts distinct codes and ignores blanks, so a
    # chemistry named twice counts once.
    tech["n_technologies"] = _codes_by_col.nunique(axis=1)

    _declared = tech["technology"].notna()
    _codes = sorted(set(_codes_by_col[_primary].dropna()))
    _unknown = [c for c in _codes if c not in TECH_NAMES]

    print("primary technology field:", _primary)
    print("codes present:", _codes)
    if _unknown:
        print("  NOT IN THE CODE TABLE:", _unknown)
        print("  These pass through under their own code and are excluded from")
        print("  the efficiency blend unless ROUND_TRIP is extended.")

    print(
        "\ngenerator-years with a technology declared:",
        int(_declared.sum()),
        "of",
        len(tech),
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
        print(
            "  None. The mix is a partition and no apportionment rule between"
            "\n  co-declared chemistries is required."
        )
    else:
        print(
            tech.loc[tech["n_technologies"] > 1,
                     ["Year", "Plant Code", "Plant Name"] + TECH_COLS]
            .to_string(index=False)
        )
    return (tech,)


@app.cell
def _(E_COL, MW_COL, tech):
    # ===== mix under three denominators =====
    # Reported together because they disagree, and because the disagreement is
    # what determines which one may be quoted. Shares are taken over the
    # declared subset, so each year sums to 100 by construction; the size of
    # that subset is reported in the cell above, not implied here.
    _t = tech[tech["technology"].notna()]

    mix = (
        _t.groupby(["Year", "technology"])
        .agg(generators=("technology", "size"), MW=(MW_COL, "sum"), MWh=(E_COL, "sum"))
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
    # The blend is anchored on the dominant chemistry and adjusted downward by
    # the remainder, so the size of that remainder is the operative number.
    #
    # The frame is reindexed over every year present in the mix. Without it a
    # year in which the reference chemistry is absent would drop out silently
    # and every downstream .loc would be reading the wrong row.
    _years = sorted(mix["Year"].unique())
    lib_share = (
        mix[mix["technology"] == REFERENCE_TECH]
        .set_index("Year")[["generators", "unit_share_%", "MW_share_%", "MWh_share_%"]]
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
    print(
        "Efficiency acts on energy throughput, so the third figure is the one"
        "\nthat weights the blend."
    )
    return (lib_share,)


@app.cell
def _(E_COL, MW_COL, PESSIMISTIC_RTE, REFERENCE_TECH, ROUND_TRIP, pd, tech):
    # ===== blended efficiency implied by the mix =====
    #
    # Conditional on the placeholder efficiencies in the parameter cell. The
    # comparison against a pure-reference fleet is the quantity of interest:
    # it states how large the downward adjustment away from the dominant
    # chemistry actually is, in efficiency points.
    #
    # The perturbation column answers the prior question. If every non-
    # reference chemistry were assumed to be PESSIMISTIC_RTE, the blend would
    # still move only as far as shown; where that movement is negligible the
    # assumed efficiencies do not need to be defended at all.
    #
    # Differences are taken before rounding. The adjustment is a fraction of an
    # efficiency point and rounding the blends first would quantise it at a
    # resolution coarser than the quantity itself.
    _t = tech[tech["technology"].notna()].copy()
    _t["rte"] = _t["technology"].map(ROUND_TRIP)
    _t["rte_pessimistic"] = _t["technology"].where(
        _t["technology"] != REFERENCE_TECH, other=None
    ).notna().map({True: PESSIMISTIC_RTE, False: ROUND_TRIP[REFERENCE_TECH]})

    _unmapped = sorted(set(_t.loc[_t["rte"].isna(), "technology"]))
    if _unmapped:
        print("technologies with no efficiency assumption, excluded:", _unmapped)
        print("  they hold",
              f"{_t.loc[_t['rte'].isna(), E_COL].sum():,.1f} MWh")

    def _wmean(frame, value, weight):
        _ok = frame[value].notna() & frame[weight].notna() & frame[weight].gt(0)
        if not _ok.any():
            return float("nan")
        return (frame.loc[_ok, value] * frame.loc[_ok, weight]).sum() / frame.loc[
            _ok, weight
        ].sum()

    _rows = []
    for _y, _g in _t.groupby("Year"):
        _g = _g[_g["rte"].notna()]
        _rows.append(
            {
                "Year": _y,
                "generators": len(_g),
                "energy_MWh": _g[E_COL].sum(),
                "blend_by_energy": _wmean(_g, "rte", E_COL),
                "blend_by_power": _wmean(_g, "rte", MW_COL),
                "blend_unweighted": _g["rte"].mean(),
                "reference_only": ROUND_TRIP[REFERENCE_TECH],
                "blend_pessimistic": _wmean(_g, "rte_pessimistic", E_COL),
            }
        )
    efficiency_blend = pd.DataFrame(_rows).set_index("Year")
    efficiency_blend["adjustment_points"] = (
        efficiency_blend["blend_by_energy"] - efficiency_blend["reference_only"]
    ) * 100
    efficiency_blend["pessimistic_points"] = (
        efficiency_blend["blend_pessimistic"] - efficiency_blend["reference_only"]
    ) * 100

    print("blended round-trip efficiency implied by the mix")
    print(f"conditional on the placeholder values in the parameter cell,"
          f" perturbation at {PESSIMISTIC_RTE}")
    print(efficiency_blend.round(4).to_string())
    print(
        "\nadjustment_points is the downward correction away from a pure"
        f"\n{REFERENCE_TECH} fleet, in efficiency points, energy weighted."
        "\npessimistic_points is the same correction if every other chemistry"
        f"\nwere {PESSIMISTIC_RTE} instead of its tabulated value."
    )
    return (efficiency_blend,)


@app.cell
def _(MW_COL, tech):
    # ===== technology heterogeneity within nodes =====
    # A single efficiency per node is defensible only where the node is
    # chemically homogeneous. Where it is not, the node parameter has to be a
    # capacity-weighted blend of its constituents rather than a lookup.
    #
    # The count of mixed nodes is the wrong summary on its own: what decides
    # whether the lookup is safe is the share of capacity sitting behind them.
    _t = tech[tech["technology"].notna()]
    _by_node = _t.groupby(["Year", "geohash"]).agg(
        technologies=("technology", "nunique"),
        generators=("technology", "size"),
        nameplate_MW=(MW_COL, "sum"),
    )
    node_tech = _by_node.reset_index()
    node_tech["mixed"] = node_tech["technologies"] > 1

    _n_mixed = int(node_tech["mixed"].sum())
    _mw_total = node_tech["nameplate_MW"].sum()
    _mw_mixed = node_tech.loc[node_tech["mixed"], "nameplate_MW"].sum()

    print("node-years with more than one technology:", _n_mixed,
          "of", len(node_tech))
    print(f"capacity at mixed nodes: {_mw_mixed:,.1f} of {_mw_total:,.1f} MW"
          f" ({_mw_mixed / _mw_total * 100:.2f}%)" if _mw_total else "no capacity")

    if _n_mixed:
        print()
        print(
            node_tech[node_tech["mixed"]]
            .sort_values(["Year", "nameplate_MW"], ascending=[False, False])
            .head(15)
            .to_string(index=False)
        )
        print(
            "\nAt these nodes a single-chemistry assumption is wrong and the"
            "\nefficiency must be blended by capacity."
        )
    else:
        print("  Every node is single-technology; one efficiency per node is safe.")
    return (node_tech,)


@app.cell
def _(ENCLOSURE_NAMES, ENC_COL, MW_COL, units):
    # ===== enclosure type =====
    # Not a model parameter. Retained because the category distribution shifts
    # between vintages in a way that looks like a reporting change rather than
    # a fleet change, which is worth having on record if the field is ever used.
    # Counts and capacity are shown together: a category that is a third of the
    # units and a twentieth of the megawatts is a different fact about the
    # fleet than the count alone suggests.
    if ENC_COL is None:
        print("no enclosure column; skipped.")
    else:
        _e = units[units[ENC_COL].notna()].copy()
        _e["enclosure"] = _e[ENC_COL].map(
            lambda v: ENCLOSURE_NAMES.get(str(v).strip().upper(), str(v).strip())
        )
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
        print(
            "\nblank enclosure cells:",
            int(units[ENC_COL].isna().sum()),
            "of",
            len(units),
        )
    return


@app.cell
def _(APP_COLS, pd, units):
    # ===== application flags, normalised once =====
    #
    # Every downstream table is derived from this frame. The parse used to be
    # repeated in three cells with three slightly different treatments of the
    # edge cases, which is exactly the arrangement in which the treatments
    # drift apart.
    #
    # Four responses are distinguished, not two. A cell that is neither Y nor N
    # nor empty is a reporting anomaly and is counted as such; folding it into
    # the blank count, as before, would inflate the apparent non-response and
    # hide the anomaly at the same time.
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
    # Three proportions are reported for each field and they answer different
    # questions. Y_share_of_answered describes respondents; Y_share_of_fleet
    # describes the fleet; response_rate says how far apart those two
    # populations are. Quoting the first without the third overstates the
    # evidence by the reciprocal of the response rate.
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
    # disagree. This is the field that bears on the four-quadrant assumption.
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
    # that decline the application while reporting a non-zero rating, or accept
    # it while reporting zero, indicate the two are populated independently,
    # which caps how far either can be relied on.
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
    # The lower row repeats the upper one with the dominant chemistry removed
    # and the axis rescaled. The remainder is what sets the efficiency
    # adjustment, and at these shares it is a line of a few pixels in the
    # full-scale panel above.
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
def _(REFERENCE_TECH, efficiency_blend, lib_share, plt):
    # ===== figure 2: reference share, and the efficiency it implies =====
    fig_lib, _axes = plt.subplots(1, 2, figsize=(11, 4.2))

    _a = _axes[0]
    for _c, _m, _col in [
        ("unit_share_%", "o", "#999999"),
        ("MW_share_%", "s", "#3c6e88"),
        ("MWh_share_%", "^", "#b4442e"),
    ]:
        _a.plot(lib_share.index, lib_share[_c], marker=_m, lw=1.8, ms=6,
                color=_col, label=_c)
    _a.set_ylim(min(60, lib_share[["unit_share_%", "MW_share_%",
                                   "MWh_share_%"]].min().min() - 5), 101)
    _a.set_xticks(list(lib_share.index))
    _a.set_ylabel("percent")
    _a.set_title(f"{REFERENCE_TECH} share by denominator", fontsize=10)
    _a.legend(fontsize=8, frameon=False, loc="lower right")
    _a.grid(alpha=0.25)

    _b = _axes[1]
    _b.plot(efficiency_blend.index, efficiency_blend["blend_by_energy"],
            marker="o", lw=1.8, ms=6, color="#b4442e", label="blend, energy weighted")
    _b.plot(efficiency_blend.index, efficiency_blend["blend_unweighted"],
            marker="v", lw=1.4, ms=5, color="#999999", label="blend, unweighted")
    _b.axhline(efficiency_blend["reference_only"].iloc[0], color="#3c6e88",
               ls="--", lw=1.2, label=f"pure {REFERENCE_TECH}")
    _b.set_xticks(list(efficiency_blend.index))
    _b.set_ylabel("round-trip efficiency")
    _b.set_title("implied blend (conditional on assumed values)", fontsize=10)
    _b.legend(fontsize=8, frameon=False)
    _b.grid(alpha=0.25)

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
    _ax.set_title(
        "A high yes-share means little where the response rate is low",
        fontsize=11,
    )
    _ax.legend(fontsize=8, frameon=False, loc="lower right")
    _ax.grid(alpha=0.2, axis="x")
    fig_apps.tight_layout()
    fig_apps
    return (fig_apps,)


@app.cell
def _(VQ_COL, plt, vq_by_year):
    # ===== figure 4: reactive support, per year and pooled =====
    # Separated from the other flags because it is the field that bears on the
    # four-quadrant assumption. Both series are plotted on one axis so that a
    # high yes-share cannot be read without the response rate beneath it.
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
    efficiency_blend,
    fig_apps,
    fig_lib,
    fig_mix,
    fig_vq,
    mix,
    node_tech,
):
    # ===== outputs =====
    mix.to_csv(PROC / "technology_mix_by_year.csv", index=False)
    efficiency_blend.reset_index().to_csv(
            PROC / "technology_efficiency_blend.csv", index=False
        )
    node_tech.to_csv(PROC / "technology_node_mixing.csv", index=False)
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
    print(f"  technology_mix_by_year.csv        {len(mix):5d} rows")
    print(f"  technology_efficiency_blend.csv   {len(efficiency_blend):5d} rows")
    print(f"  technology_node_mixing.csv        {len(node_tech):5d} rows")
    print(f"  applications_summary.csv          {len(apps):5d} rows")
    print(f"  applications_by_year.csv          {len(apps_by_year):5d} rows")
    print(f"  figures                           {3 + (fig_vq is not None)}")
    print()
    print("technology_efficiency_blend.csv supplies the model input:")
    print("  blend_by_energy     round-trip efficiency for the fleet")
    print("  adjustment_points   its distance from a single-chemistry value")
    print("technology_node_mixing.csv flags the nodes at which one efficiency")
    print("  per node is not adequate.")
    return


@app.cell
def _(
    E_COL,
    PESSIMISTIC_RTE,
    Q_COL,
    REFERENCE_TECH,
    VQ_COL,
    apps,
    efficiency_blend,
    lib_share,
    node_tech,
    tech,
    units,
    vq_by_year,
    vq_disagree,
):
    # ===== findings =====
    # Every number is read from the current run, and every comparative claim is
    # decided by a test on those numbers rather than asserted, so a sentence
    # that would be false under a different input is not printed under that
    # input.
    _first = int(lib_share.index.min())
    _last = int(lib_share.index.max())

    print("=" * 70)
    print(f"FINDINGS   technology and applications, {_first}-{_last}")
    print("=" * 70)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    print("\n2. Coverage of the subset the blend is computed over.")
    _dec = tech["technology"].notna()
    _cov_units = _dec.mean() * 100
    _cov_energy = (
        tech.loc[_dec, E_COL].sum() / tech[E_COL].sum() * 100
        if tech[E_COL].sum() else float("nan")
    )
    print(f"   generator-years declaring a chemistry: {_cov_units:.1f}%")
    print(f"   the energy they hold                 : {_cov_energy:.2f}%")
    if _cov_energy >= 99:
        print("   The declared subset is the fleet in all but name, so the blend")
        print("   may be quoted as a fleet parameter without qualification.")
    else:
        print("   A material share of fleet energy carries no declared chemistry.")
        print("   The blend describes the declared subset and inherits whatever")
        print("   the undeclared remainder turns out to be.")

    # ------------------------------------------------------------------
    print("\n3. Efficiency adjustment implied by the remainder.")
    _adj = efficiency_blend.loc[_last, "adjustment_points"]
    _pess = efficiency_blend.loc[_last, "pessimistic_points"]
    print(f"   energy-weighted blend"
          f" {efficiency_blend.loc[_last, 'blend_by_energy']:.4f} against"
          f" {efficiency_blend.loc[_last, 'reference_only']:.4f} for a pure fleet")
    print(f"   adjustment {_adj:+.3f} efficiency points")
    print(f"   with every other chemistry at {PESSIMISTIC_RTE}:"
          f" {_pess:+.3f} points")
    if abs(_pess) < 0.5:
        print("   Even under that assumption the correction is below half a")
        print("   point, so the assumed efficiencies need not be defended and")
        print(f"   a single {REFERENCE_TECH} value is adequate for the fleet.")
    else:
        print("   The correction exceeds half a point under the perturbation, so")
        print("   the assumed efficiencies must be sourced before the blend is")
        print("   used; the placeholder table is load bearing.")

    # ------------------------------------------------------------------
    print("\n4. Whether one efficiency per node is defensible.")
    _multi = int((tech["n_technologies"] > 1).sum())
    _nmix = int(node_tech["mixed"].sum())
    _mw_mixed = node_tech.loc[node_tech["mixed"], "nameplate_MW"].sum()
    _mw_all = node_tech["nameplate_MW"].sum()
    _mw_share = _mw_mixed / _mw_all * 100 if _mw_all else float("nan")
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
        print("   Every node is single-technology: a per-node efficiency lookup")
        print("   is exact.")
    elif _mw_share < 5:
        print("   Mixed nodes exist but hold little capacity, so a per-node")
        print("   lookup is a small and bounded approximation.")
    else:
        print("   Mixed nodes hold a substantial share of capacity, so the node")
        print("   efficiency must be a capacity-weighted blend, not a lookup.")

    # ------------------------------------------------------------------
    print("\n5. Application fields are sparsely answered.")
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

    # ------------------------------------------------------------------
    if VQ_COL is not None and len(vq_by_year):
        _tot = int(vq_by_year["total"].sum())
        _ans = int(vq_by_year["answered"].sum())
        _yes = int(vq_by_year["Y"].sum())
        print("\n6. Four-quadrant operation cannot be settled by the flag.")
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
            print("   The ratings carry the burden of evidence on capability; this")
            print("   field can corroborate but cannot establish it.")
        else:
            print("   The flag is answered at least as often as the rating is")
            print("   reported, so it is usable as independent evidence.")
    print("=" * 70)
    return


if __name__ == "__main__":
    app.run()
