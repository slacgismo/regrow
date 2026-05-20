# PySAM Wind Power Modeling Pipeline

Simulates hourly wind turbine power output for individual turbines in the US Wind Turbine Database (USWTDB) using PySAM's Windpower model and wind resource data from NREL's Wind Toolkit (WTK BCHRRR v1.0.0). Results are written to S3 per turbine.

## Overview

For each turbine in `uswtdb_metadata.csv`, the pipeline:

1. Filters and imputes missing turbine metadata using the most common turbine model in the dataset
2. Fetches hourly wind resource data (wind speed, direction, temperature, pressure) from the NREL WTK API at the closest available hub height
3. Converts weather data to SAM Resource Wind (SRW) format
4. Runs PySAM's single-turbine Windpower model year by year (8760-hour chunks)
5. Saves power output time series and raw weather data to S3, and a power curve plot locally

## Dependencies

```
PySAM
pandas
numpy
matplotlib
requests
```

Install via conda:
```bash
conda install -c conda-forge pysam pandas numpy matplotlib requests
```

Also requires the following from `utils.py`:

- `geohash(lat, lon, precision)` — generates a geohash string used in file naming
- `nsrdb_credentials()` — returns `(email, api_key)` for NREL API access
- `nsrdb_weather(...)` — NSRDB weather fetch utility

## Configuration

**NREL API credentials** are loaded from `utils.nsrdb_credentials()`. Ensure your NREL developer API key is configured. Keys are available at https://developer.nrel.gov/signup/.

**Input file:** `uswtdb_metadata.csv` must be present in the working directory.

Required columns:

| Column | Description |
|---|---|
| `bus` | WECC bus identifier |
| `latitude` / `longitude` | Turbine coordinates |
| `capacity[MW]` | Turbine nameplate capacity |
| `hub_height[m]` | Hub height in meters |
| `rotor_diameter[m]` | Rotor diameter in meters |
| `cut_in_wind_speed[m/s]` | Cut-in wind speed |
| `rated_wind_speed[m/s]` | Rated wind speed |
| `cut_out_wind_speed[m/s]` | Cut-out wind speed |
| `max_rotor_speed[rd/min]` | Maximum rotor speed |
| `drivetrain_design` | Drivetrain type string (see below) |
| `gentype` | Generator type (used for imputation) |
| `year` | Year the turbine came online |
| `elevation[m]` | Site elevation |
| `county` | County name (last 2 chars used as state code) |

**Output directory:** `./pysam_wecc_nodes/` must exist with the following subdirectories before running:

```bash
mkdir -p pysam_wecc_nodes/srw_files
mkdir -p pysam_wecc_nodes/plots
```

**S3 output path:** Set `regrow_folder` and `aws_profile` at the top of `__main__`:

```python
regrow_folder = "s3://pvdrdb-transfer/REGROW/pysam_wind_powerplants/"
aws_profile   = "aws-service-creds-pvdrdb"
```

## Usage

```bash
python pysam_wind_pipeline.py
```

The pipeline loops through all rows in `uswtdb_metadata.csv` sorted by turbine count per bus (ascending). Turbines that already have a plot in `pysam_wecc_nodes/plots/` are skipped — the pipeline is safe to re-run after interruption.

A filtered and imputed copy of the metadata is written to `uswtdb_pysam_sim.csv` at startup for reference.

## Model Details

### Metadata Imputation

Turbines with any missing required parameter are replaced wholesale with the metadata of the most common turbine `gentype` in the dataset. This applies to all required fields listed above except `drivetrain_design`.

### Date Range

- Turbines online in 2018 or earlier: simulated from 2018-01-01 through 2022-12-30
- Turbines online after 2018: simulated from January 1 of their online year through 2022-12-30
- Turbines with online year after 2022 or missing year are excluded

### Wind Resource Data

Wind data is fetched from the **NREL WTK BCHRRR v1.0.0** API for CONUS sites. The hub height is snapped to the nearest available WTK level:

```
Available heights: 10m, 20m, 40m, 60m, 80m, 100m, 120m, 140m, 160m, 180m
```

Retrieved variables at the snapped height: wind speed, wind direction, temperature, surface pressure (0m).

A 1-second sleep is added between annual API calls to avoid rate limiting.

### SRW Conversion

Weather data is converted to SAM Resource Wind (SRW) format before being passed to PySAM. Pressure is converted from Pa to atm. The SRW file is written to disk temporarily and removed after each model run.

### Power Curve

The power curve is calculated from turbine metadata using `wm.Turbine.calculate_powercurve()` with:

| Parameter | Source |
|---|---|
| Turbine size | `capacity[MW]` × 1000 (kW) |
| Rotor diameter | `rotor_diameter[m]` |
| Elevation | `elevation[m]` + `hub_height[m]` |
| Max Cp | 0.47 (fixed average) |
| Max tip speed | Derived from `max_rotor_speed` and rotor diameter |
| Max tip speed ratio | Rounded ratio of max tip speed to rated wind speed |
| Cut-in / cut-out | From metadata |
| Drive train | Integer mapped from `drivetrain_design` string (see below) |

**Drivetrain mapping:**

| String | Integer |
|---|---|
| `Single Stage - Low Speed Generator` | 1 |
| `Multi-Generator` | 2 |
| `Direct Drive` | 3 |
| All others (default) | 0 — 3 Stage Planetary |

### Model Parameters

- Wake model: disabled (`wind_farm_wake_model = 0`)
- Resource model: hourly (`wind_resource_model_choice = 0`)
- Turbulence coefficient: 0.10
- Shear exponent: 0.143 (neutral atmosphere)
- Turbine layout: single turbine at origin (multi-turbine runs space turbines 500m apart on x-axis)
- Simulation chunk size: 8760 hours (one calendar year); incomplete chunks are skipped

## Output Files

**Power time series** (per turbine, written to S3):
```
s3://pvdrdb-transfer/REGROW/pysam_wind_powerplants/single_turbine_power_timeseries/{bus}_{lat}_{lon}.csv
```

| Column | Description |
|---|---|
| `datetime` | Hourly UTC timestamps |
| `power[kW]` | Modeled power output in kW |

**Raw weather data** (per turbine, written to S3):
```
s3://pvdrdb-transfer/REGROW/pysam_wind_powerplants/single_turbine_weather_data/{bus}_{lat}_{lon}.csv
```

Raw WTK API response including wind speed, direction, temperature, and pressure columns at the snapped hub height.

**Power curve plots** (local):
```
pysam_wecc_nodes/plots/{bus}_{lat}_{lon}.png
```

Time series plot of modeled power output across all simulated years. Used as the completion flag — if this file exists the turbine is skipped on re-run.

## Notes

- Capacity is converted from MW to kW before being passed to PySAM
- The `HOME` environment variable is overridden at the top of the script (`os.environ["HOME"] = "C:/users/kperry"`) — update this path if running on a different machine or on Kestrel
- Incomplete year chunks (fewer than 8760 hours) are silently skipped — this affects the final year if `max_measured_date` does not fall on a full year boundary
- `plot_powercurve=True` by default in `run_single_turbine_pysam_model` — this will display a blocking plot window for every turbine run; set to `False` for unattended batch runs

---

## Step 2: Aggregating to WECC Node Level

After per-turbine CSVs are written to S3, run `agg-powerplant-predictions-node-wind.py` to aggregate turbine-level output up to WECC bus nodes and produce a final geopanel file.

```bash
python agg-powerplant-predictions-node-wind.py
```

### What It Does

The script runs in two passes:

**Pass 1 — Aggregate turbines to bus nodes:**
For each unique WECC bus in `uswtdb.csv`, it finds all matching turbine CSVs on S3 by matching `{lat}_{lon}.csv` filenames against the metadata. It reads each file directly from S3, aligns on the time index (deduplicating any repeated timestamps), and sums output into a `sum_pp` column. One CSV is written per bus to the local `pysam_wind_bus_agg/` directory.

**Pass 2 — Build the geopanel:**
Reads all bus-level CSVs from `pysam_wind_bus_agg/`, extracts `sum_pp` for each bus (rounded to 2 decimal places), converts the index to UTC, and concatenates into a wide-format geopanel with one column per WECC bus. Written to `pysam_geopanel.csv`.

### Configuration

Update the following variables at the top of the script before running:

| Variable | Default | Description |
|---|---|---|
| `aws_profile` | `991404956194_regrow-developer` | AWS credentials profile for S3 access |
| `base_path` | `C:/Users/kperry/.../pysam_wecc_nodes` | Local root directory for outputs |
| `power_plant_path` | `regrow/pysam_wind_powerplants/single_turbine_power_timeseries/` | S3 path prefix for per-turbine CSVs |
| `aggregated_pp_wecc_node_path` | `pysam_wind_bus_agg` | Local subdirectory for per-bus aggregated outputs |
| `geopanel_file_path` | `pysam_geopanel.csv` | Local path for final geopanel output |
| `metadata_path` | `uswtdb.csv` | USWTDB metadata file (same as Step 1) |

The `HOME` environment variable is overridden at the top — update `os.environ["HOME"]` to match your machine or remove it on Kestrel.

The local output directory must exist before running:
```bash
mkdir -p pysam_wecc_nodes/pysam_wind_bus_agg
```

Per-bus files that have already been written can be skipped by uncommenting the `already_run_files` check in the loop.

### Output Files

**Per-bus CSVs** (local, `pysam_wind_bus_agg/{bus}.csv`):

| Column | Description |
|---|---|
| timestamp (index) | Hourly UTC timestamps |
| `{lat}_{lon}` | Power output (kW) for each individual turbine at the bus |
| `sum_pp` | Total aggregated power output (kW) across all turbines at the bus |

**Geopanel** (local, `pysam_geopanel.csv`):

Wide-format file with one column per WECC bus, rows as hourly UTC timestamps, values as total bus-level power output in kW rounded to 2 decimal places. This is the primary input for downstream node-level wind analysis.

### Differences from the Solar Aggregation Script

This script differs from the PVWatts equivalent (`agg-powerplant-predictions-node.py`) in three key ways:

- **Source files are on S3**, not local disk — listed via `s3fs` and read directly with `storage_options`
- **Turbine files are matched by `{lat}_{lon}.csv`** rather than by bus prefix in the filename
- **Duplicate timestamp deduplication** is applied per turbine before concatenation, and the geopanel index is explicitly converted to UTC

### Full Pipeline Order

```
1. python pysam_wind_pipeline.py
        ↓ writes to S3:
            single_turbine_power_timeseries/{bus}_{lat}_{lon}.csv
            single_turbine_weather_data/{bus}_{lat}_{lon}.csv

2. python agg-powerplant-predictions-node-wind.py
        ↓ writes pysam_wecc_nodes/pysam_wind_bus_agg/{bus}.csv
        ↓ writes pysam_geopanel.csv
```
