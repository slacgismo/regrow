# -*- coding: utf-8 -*-
"""
Run PVWatts for DG

Updates 6/5/26:
1) Fixes silent failures: no bare except; logs errors and raises if weather missing
2) Caches NSRDB API calls to disk per (geohash, year, time_step)
3) Refactors settings and filenames to global variables at the top

Modeling steps intentionally unchanged inside run_pvwatts_model().
"""

from __future__ import annotations

import time
from pathlib import Path
import pandas as pd
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import utils


# ----------------------------
# Global settings / filenames
# ----------------------------

# Input/output files
METADATA_CSV = "wecc_bus_dg_cap_and_gen_by_month.csv"
OUTPUT_CSV = "residential_solar_geopanel5.csv"

# NSRDB / caching
CACHE_DIR = Path("nsrdb_cache")
NSRDB_START_YEAR = 2018
NSRDB_END_YEAR_INCLUSIVE = 2022
NSRDB_TIME_STEP_MIN = 30

NSRDB_RETRIES = 3
NSRDB_BACKOFF_BASE_SECONDS = 2  # exponential base (1,2,4,...)

# PV / model settings (keep consistent with original)
TILT_DEG = 20
AZIMUTH_DEG = 180
TRACKING = False

ALBEDO = 0.2
IRRADIANCE_MODEL = "perez"
IAM_N = 1.5

TEMP_MODEL_FAMILY = "sapm"
TEMP_MODEL_KEY = "open_rack_glass_glass"
TEMP_COEFFICIENT = -0.0047  # gamma_pdc

# Metadata column names
COL_GEOHASH = "geohash"
COL_LAT = "lat"
COL_LON = "lon"
COL_YEAR = "Year"
COL_MONTH = "Month"
COL_CAP_MW = "Capacity [MW]"


def run_pvwatts_model(
    tilt, azimuth, dc_capacity, dc_inverter_limit,
    solar_zenith, solar_azimuth, dni, dhi, ghi, dni_extra,
    relative_airmass, temperature, wind_speed,
    temperature_model_parameters,
    temperature_coefficient, tracking
):
    """
    Run the PVWatts model using NSRDB data across the time period as inputs.
    Modeling steps identical to K.P.'s original script.
    """
    if tracking:
        tracker_angles = pvlib.tracking.singleaxis(
            solar_zenith, solar_azimuth,
            axis_tilt=tilt, axis_azimuth=azimuth,
            backtrack=True, gcr=0.4, max_angle=60
        )
        surface_tilt = tracker_angles["surface_tilt"]
        surface_azimuth = tracker_angles["surface_azimuth"]
    else:
        surface_tilt = tilt
        surface_azimuth = azimuth

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth,
        solar_zenith, solar_azimuth,
        dni, ghi, dhi,
        dni_extra=dni_extra,
        airmass=relative_airmass,
        albedo=ALBEDO,
        model=IRRADIANCE_MODEL
    )

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)

    # IAM
    iam = pvlib.iam.physical(aoi, n=IAM_N)

    # Apply IAM to direct POA only
    poa_transmitted = poa["poa_direct"] * iam + poa["poa_diffuse"]

    temp_cell = pvlib.temperature.sapm_cell(
        poa["poa_global"],
        temperature,
        wind_speed,
        **temperature_model_parameters
    )

    pdc = pvlib.pvsystem.pvwatts_dc(
        poa_transmitted,
        temp_cell,
        dc_capacity,
        temperature_coefficient
    )
    return pdc


def _validate_metadata(metadata: pd.DataFrame) -> None:
    required = {COL_GEOHASH, COL_LAT, COL_LON, COL_YEAR, COL_MONTH, COL_CAP_MW}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")


def _cache_path_for(geohash: str, year: int) -> Path:
    # Include time_step in cache key so cache stays correct if you change it later
    safe_geohash = str(geohash).replace(os.sep if "os" in globals() else "/", "_")
    return CACHE_DIR / f"nsrdb_{safe_geohash}_{year}_t{NSRDB_TIME_STEP_MIN}.parquet"


def fetch_nsrdb_year_cached(
    *,
    geohash: str,
    lat: float,
    lon: float,
    year: int,
    email: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Fetch NSRDB for a single year with disk cache + retries + visible failures.
    Returns a DataFrame with a tz-aware DatetimeIndex (as provided by pvlib).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for(geohash, year)

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    last_err = None
    for attempt in range(1, NSRDB_RETRIES + 1):
        try:
            df, meta = pvlib.iotools.get_nsrdb_psm4_conus(
                latitude=lat,
                longitude=lon,
                api_key=api_key,
                email=email,
                year=year,
                map_variables=True,
                time_step=NSRDB_TIME_STEP_MIN,
            )

            needed = {"dni", "dhi", "ghi", "temp_air", "wind_speed"}
            missing = needed - set(df.columns)
            if missing:
                raise ValueError(f"NSRDB returned missing columns {sorted(missing)} for {geohash} {year}")

            # Basic index hygiene (no modeling change)
            df = df[~df.index.duplicated(keep="first")].sort_index()

            df.to_parquet(cache_path)
            return df

        except Exception as e:
            last_err = e
            sleep_s = NSRDB_BACKOFF_BASE_SECONDS ** (attempt - 1)
            print(f"WARNING: NSRDB fetch failed for geohash={geohash} year={year} "
                  f"(attempt {attempt}/{NSRDB_RETRIES}): {e}")
            time.sleep(sleep_s)

    raise RuntimeError(f"NSRDB fetch failed for geohash={geohash} year={year}") from last_err


def month_bounds(index: pd.DatetimeIndex, year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if index.tz is None:
        raise ValueError("Weather index is not timezone-aware; cannot safely tz_localize month boundaries.")
    start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0, second=0, tz=index.tz)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def main() -> None:
    metadata = pd.read_csv(METADATA_CSV)
    _validate_metadata(metadata)

    email, api_key = utils.nsrdb_credentials()

    geohash_frames: list[pd.DataFrame] = []

    for geohash in metadata[COL_GEOHASH].drop_duplicates():
        print(f"Running monthly production for geohash {geohash}...")

        metadata_subset = metadata[metadata[COL_GEOHASH] == geohash]
        site_lat = float(metadata_subset[COL_LAT].iloc[0])
        site_lon = float(metadata_subset[COL_LON].iloc[0])

        # Fetch weather (cached) for all configured years
        weather_years = []
        for year in range(NSRDB_START_YEAR, NSRDB_END_YEAR_INCLUSIVE + 1):
            dfy = fetch_nsrdb_year_cached(
                geohash=str(geohash),
                lat=site_lat,
                lon=site_lon,
                year=year,
                email=email,
                api_key=api_key,
            )
            weather_years.append(dfy)

        master_weather_df = pd.concat(weather_years)
        master_weather_df = master_weather_df[~master_weather_df.index.duplicated(keep="first")].sort_index()

        if master_weather_df.empty:
            raise RuntimeError(f"Empty master_weather_df for geohash={geohash}; cannot proceed.")

        temp_params = TEMPERATURE_MODEL_PARAMETERS[TEMP_MODEL_FAMILY][TEMP_MODEL_KEY]

        # Month-by-month modeling (same as your current structure)
        agg_df_list: list[pd.DataFrame] = []
        for _, row in metadata_subset.iterrows():
            year = int(row[COL_YEAR])
            month = int(row[COL_MONTH])

            # Capacity [MW] -> kW (same as your script)
            power_kw = float(row[COL_CAP_MW]) * 1000.0

            start, end = month_bounds(master_weather_df.index, year, month)
            weather_subset_df = master_weather_df.loc[(master_weather_df.index >= start) & (master_weather_df.index < end)]

            if weather_subset_df.empty:
                print(f"WARNING: No weather data for geohash={geohash} {year}-{month:02d}; skipping month.")
                continue

            solpos = pvlib.solarposition.get_solarposition(weather_subset_df.index, site_lat, site_lon)
            dni_extra = pvlib.irradiance.get_extra_radiation(weather_subset_df.index)
            relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)

            pdc = run_pvwatts_model(
                tilt=TILT_DEG,
                azimuth=AZIMUTH_DEG,
                dc_capacity=power_kw,
                dc_inverter_limit=power_kw * 1.5,  # still unused; kept for parity
                solar_zenith=solpos.zenith,
                solar_azimuth=solpos.azimuth,
                dni=weather_subset_df["dni"],
                dhi=weather_subset_df["dhi"],
                ghi=weather_subset_df["ghi"],
                dni_extra=dni_extra,
                relative_airmass=relative_airmass,
                temperature=weather_subset_df["temp_air"],
                wind_speed=weather_subset_df["wind_speed"],
                temperature_model_parameters=temp_params,
                temperature_coefficient=TEMP_COEFFICIENT,
                tracking=TRACKING,
            )

            pdc.name = str(geohash)
            agg_df_list.append(pd.DataFrame(pdc))

        if not agg_df_list:
            print(f"WARNING: No modeled months produced for geohash={geohash}; skipping geohash.")
            continue

        node_production = pd.concat(agg_df_list)
        node_production = node_production[~node_production.index.duplicated(keep="first")].sort_index()

        geohash_frames.append(node_production)

        # Optional: progressive write (keeps your current behavior of writing each loop)
        geohash_output = pd.concat(geohash_frames, axis=1).sort_index()
        geohash_output.to_csv(OUTPUT_CSV)

    if not geohash_frames:
        raise RuntimeError("No geohash outputs produced.")

    # Final write (ensures complete file even if you disable progressive writes)
    geohash_output = pd.concat(geohash_frames, axis=1).sort_index()
    geohash_output.to_csv(OUTPUT_CSV)
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()