#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------
# GLOBAL PATHS (edit these)
# ----------------------------
OUTPUT_CSV = "residential_solar_geopanel_TEST02.csv"          # pvlib time series, timestamp stored as index
METADATA_CSV = "wecc_bus_dg_cap_and_gen_by_month.csv"            # monthly actuals with Generation [MWh]

SCALING_FACTORS_CSV = "scaling_factors.csv"
CORRECTED_TIMESERIES_CSV = "residential_solar_geopanel_corrected.csv"  # wide, same shape as input

# ----------------------------
# CONFIG (matches your samples)
# ----------------------------
GEN_COL = "Generation [MWh]"
META_YEAR_COL = "Year"
META_MONTH_COL = "Month"
META_LOC_COL = "geohash"

# Simulated power units in OUTPUT_CSV: one of {"W","kW","MW"}
POWER_UNIT = "kW"   # change if needed


def _power_to_mwh(power_values: pd.Series, dt_hours: float) -> pd.Series:
    if POWER_UNIT == "W":
        return power_values * (dt_hours / 1e6)
    if POWER_UNIT == "kW":
        return power_values * (dt_hours / 1e3)
    if POWER_UNIT == "MW":
        return power_values * dt_hours
    raise ValueError(f"Unsupported POWER_UNIT={POWER_UNIT}")


def _parse_metadata_monthly(meta: pd.DataFrame) -> pd.DataFrame:
    for c in [META_LOC_COL, META_YEAR_COL, META_MONTH_COL, GEN_COL]:
        if c not in meta.columns:
            raise ValueError(f"Metadata missing required column: {c}")

    out = meta.copy()
    out["loc_key"] = out[META_LOC_COL].astype(str)

    out["_date"] = pd.to_datetime(
        dict(year=out[META_YEAR_COL].astype(int), month=out[META_MONTH_COL].astype(int), day=1)
    )
    out["month"] = out["_date"].dt.to_period("M")
    out[GEN_COL] = pd.to_numeric(out[GEN_COL], errors="coerce")

    return (
        out.groupby(["loc_key", "month"], as_index=False)[GEN_COL]
        .sum(min_count=1)
        .rename(columns={GEN_COL: "generation_mwh_actual"})
    )


def _load_sim_timeseries_indexed(path: str) -> pd.DataFrame:
    """
    Reads a wide pvlib output where timestamp was written as the CSV index.
    Returns a DataFrame with a DatetimeIndex and columns=geohash.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def _infer_dt_hours(dt_index: pd.DatetimeIndex) -> float:
    diffs = pd.Series(dt_index).diff().dropna().dt.total_seconds() / 3600.0
    if len(diffs) == 0:
        raise ValueError("Time series has <2 rows; cannot infer timestep.")
    dt_hours = float(diffs.median())
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError(f"Invalid inferred timestep (hours): {dt_hours}")
    return dt_hours


def _compute_monthly_sim_mwh(sim_wide: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
    """
    Returns long table: loc_key, month, generation_mwh_sim
    Treats missing power values as 0.0 for integration.
    """
    sim = sim_wide.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # long: timestamp, loc_key, power
    long_ts = (
        sim.reset_index(names="timestamp")
        .melt(id_vars=["timestamp"], var_name="loc_key", value_name="power_sim")
    )
    long_ts["month"] = long_ts["timestamp"].dt.to_period("M")

    long_ts["energy_mwh_sim_step"] = _power_to_mwh(long_ts["power_sim"], dt_hours)

    monthly = (
        long_ts.groupby(["loc_key", "month"], as_index=False)["energy_mwh_sim_step"]
        .sum(min_count=1)
        .rename(columns={"energy_mwh_sim_step": "generation_mwh_sim"})
    )
    return monthly


def _compute_scaling_factors(monthly_sim: pd.DataFrame, monthly_actual: pd.DataFrame) -> pd.DataFrame:
    merged = monthly_sim.merge(monthly_actual, on=["loc_key", "month"], how="outer")

    sim = merged["generation_mwh_sim"]
    act = merged["generation_mwh_actual"]

    merged["scaling_factor"] = np.where(
        sim.isna() | act.isna(),
        np.nan,
        np.where(
            (sim == 0) & (act == 0),
            1.0,
            np.where(
                (sim == 0) & (act != 0),
                np.nan,          # cannot scale from 0 to nonzero
                act / sim,
            ),
        ),
    )

    merged["generation_mwh_sim_scaled"] = merged["generation_mwh_sim"] * merged["scaling_factor"]
    merged["abs_error_mwh_after"] = (merged["generation_mwh_actual"] - merged["generation_mwh_sim_scaled"]).abs()

    return merged.sort_values(["loc_key", "month"])


def _apply_monthly_factors_to_wide(sim_wide: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Applies per-(geohash, month) scaling to each timestamp row and returns wide DF
    with the same index/columns as sim_wide.

    Any missing scaling_factor becomes NaN in output for that month+geohash.
    """
    # factors -> wide table of scaling factors with index=month, columns=loc_key
    f = factors[["loc_key", "month", "scaling_factor"]].copy()
    f["month"] = f["month"].astype(str)  # for pivot stability
    factor_wide = f.pivot(index="month", columns="loc_key", values="scaling_factor")

    # map each timestamp to its month key string
    month_key = sim_wide.index.to_period("M").astype(str)

    # build a factor matrix aligned to sim_wide rows/cols
    # (reindex to ensure same geohash column set/order)
    factor_for_rows = factor_wide.reindex(index=pd.Index(month_key, name="month"))
    factor_for_rows = factor_for_rows.reindex(columns=sim_wide.columns)

    # Apply
    sim_num = sim_wide.copy().apply(pd.to_numeric, errors="coerce")
    corrected = sim_num.to_numpy() * factor_for_rows.to_numpy()

    corrected_wide = pd.DataFrame(corrected, index=sim_wide.index, columns=sim_wide.columns)
    return corrected_wide


def main() -> None:
    meta_raw = pd.read_csv(METADATA_CSV)
    sim_wide = _load_sim_timeseries_indexed(OUTPUT_CSV)

    dt_hours = _infer_dt_hours(sim_wide.index)

    meta_monthly = _parse_metadata_monthly(meta_raw)
    monthly_sim = _compute_monthly_sim_mwh(sim_wide, dt_hours)
    factors = _compute_scaling_factors(monthly_sim, meta_monthly)

    # (1) scaling factors file
    factors_out = factors[
        ["loc_key", "month", "generation_mwh_sim", "generation_mwh_actual", "scaling_factor", "abs_error_mwh_after"]
    ].copy()
    factors_out["month"] = factors_out["month"].astype(str)
    factors_out.to_csv(SCALING_FACTORS_CSV, index=False)

    # (2) corrected time series in SAME SHAPE as input (timestamp index + geohash columns)
    corrected_wide = _apply_monthly_factors_to_wide(sim_wide, factors)
    corrected_wide.to_csv(CORRECTED_TIMESERIES_CSV, index=True)

    print(f"Inferred timestep: {dt_hours:.6g} hours")
    print(f"Wrote: {SCALING_FACTORS_CSV}")
    print(f"Wrote: {CORRECTED_TIMESERIES_CSV}")


if __name__ == "__main__":
    main()