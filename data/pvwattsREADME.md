# PVWatts Solar Power Modeling Pipeline

Simulates DC power output for utility-scale PV systems using NSRDB weather data and the PVWatts model via pvlib. Designed to generate modeled generation baselines for solar plants in the US PV Database (USPVDB) from 2018 through 2022.

## Overview

For each plant in `uspvdb.csv`, the pipeline:

1. Fetches hourly NSRDB PSM4 weather data (DNI, DHI, GHI, temperature, wind speed) for each year the plant was operational
2. Computes solar position, airmass, and extraterrestrial radiation
3. Runs a PVWatts DC model with Perez POA transposition, physical IAM correction, and SAPM cell temperature
4. Writes modeled DC output (kW, hourly) to a CSV file per plant

Output files are written locally to the `pvwatts_powerplants/` directory.

## Dependencies

```
pvlib
pandas
numpy
matplotlib
requests
```

Install via conda:
```bash
conda install -c conda-forge pvlib pandas numpy matplotlib requests
```

Also requires two internal utilities from `utils.py`:

- `geohash(lat, lon, precision)` — generates a geohash string used in file naming
- `nsrdb_credentials()` — returns `(email, api_key)` for NSRDB API access

## Configuration

**NSRDB API credentials** are loaded from `utils.nsrdb_credentials()`. Ensure your NSRDB API key is configured before running. Keys are available at https://developer.nrel.gov/signup/.

**Input file:** `uspvdb.csv` must be present in the working directory.

Required columns:

| Column | Description |
|---|---|
| `latitude` | Plant latitude |
| `longitude` | Plant longitude |
| `name` | Plant name |
| `bus` | Bus identifier (used in output filename) |
| `capacity[MW]` | Nameplate DC capacity in MW |
| `tilt[deg]` | Panel tilt angle in degrees |
| `azimuth[deg]` | Panel azimuth angle in degrees |
| `axis` | Mounting type: `FIXED_TILT` or other (treated as single-axis tracking) |
| `year` | Year the plant came online |

**Output directory:** Set `data_path` at the top of `__main__` to your desired local output folder. Default:
```
C:/Users/kperry/Documents/extreme-weather-ca-heatwave/pvwatts_powerplants
```

## Usage

```bash
python pvwatts_pipeline.py
```

The pipeline loops through all rows in `uspvdb.csv`. Plants that already have an output CSV in `data_path` can be skipped by uncommenting the `already_run` check near the top of the loop.

## Model Details

### Irradiance Transposition

POA irradiance is computed using the **Perez model** via `pvlib.irradiance.get_total_irradiance()` with a fixed ground albedo of 0.2.

### Tracking

- `FIXED_TILT` — uses the tilt and azimuth from the metadata directly
- All other axis types — treated as **single-axis tracking** with `pvlib.tracking.singleaxis()`, using GCR=0.4, max rotation angle of 60°, and backtracking enabled

If tilt is missing (`nan`): defaults to 0° for tracking systems and 20° for fixed-tilt.

### IAM Correction

Incidence angle modifier is applied using the **physical IAM model** (`pvlib.iam.physical`) with refractive index n=1.5. IAM is applied to the direct POA component only; diffuse is passed through unchanged.

### Cell Temperature

Cell temperature is modeled using **SAPM** (`pvlib.temperature.sapm_cell`) with the `open_rack_glass_glass` parameter set.

### DC Output

DC power is computed using **PVWatts** (`pvlib.pvsystem.pvwatts_dc`) with:
- Temperature coefficient: -0.0047 /°C
- DC capacity: plant nameplate in kW (converted from MW)

## Output Format

One CSV file per plant, named:

```
{bus}_{name}_{lat}_{lon}.csv
```

Spaces and forward slashes in bus/name are replaced with underscores.

**Columns:**

| Column | Description |
|---|---|
| timestamp (index) | Hourly UTC timestamps |
| `output_kW` | Modeled DC power output in kW |

## Date Range

- Simulation starts at 2018-01-01 or the plant's online year (whichever is later)
- Simulation ends at 2023-01-01 for all plants
- Plants that came online after 2022 are effectively excluded (no data range)

## Error Handling

NSRDB API calls include 3 retry attempts per year with silent failure on exception. If all retries fail for a given year, that year is simply absent from the weather data — no error is raised. Check output file length if completeness matters.

## Notes

- Capacity is converted from MW to kW before being passed to the model
- The `dc_inverter_limit` is set to `1.5 × dc_capacity` (DC/AC ratio of 1.5), consistent with utility-scale PV norms, but this value is not enforced as an AC clipping limit in the current implementation — only DC output is returned
- A plot of modeled output is displayed for each plant during the run; close the window to continue to the next plant

---

## Step 2: Aggregating to WECC Node Level

After the per-plant CSVs are generated, run `agg-powerplant-predictions-node.py` to aggregate plant-level output up to WECC bus nodes and produce a final geopanel file.

```bash
python agg-powerplant-predictions-node.py
```

### What It Does

The script runs in two passes:

**Pass 1 — Aggregate plants to bus nodes:**
For each unique WECC bus in `uspvdb.csv`, it finds all matching plant CSVs in `pvwatts_powerplants/` (matched by filename prefix), aligns them on their time index, and sums them into a single `sum_pp` column. Each plant column is labeled by its geohash. One CSV is written per bus to `pvwatts_bus_agg/`.

**Pass 2 — Build the geopanel:**
Reads all bus-level CSVs from `pvwatts_bus_agg/`, extracts the `sum_pp` column for each, and concatenates them into a wide-format geopanel with one column per WECC bus. Written to `pvwatts_geopanel.csv`.

### Configuration

All paths are set at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `base_path` | `C:/Users/kperry/Documents/extreme-weather-ca-heatwave` | Root directory |
| `power_plant_path` | `pvwatts_powerplants` | Per-plant CSV inputs (output of Step 1) |
| `aggregated_pp_wecc_node_path` | `pvwatts_bus_agg` | Per-bus aggregated outputs |
| `geopanel_file_path` | `pvwatts_geopanel.csv` | Final geopanel output |
| `metadata_path` | `uspvdb.csv` | Same metadata file used in Step 1 |

The `pvwatts_bus_agg/` directory must exist before running — create it if needed:
```bash
mkdir C:/Users/kperry/Documents/extreme-weather-ca-heatwave/pvwatts_bus_agg
```

### Output Files

**Per-bus CSVs** (`pvwatts_bus_agg/{bus}.csv`):

| Column | Description |
|---|---|
| timestamp (index) | Hourly UTC timestamps |
| `{geohash}` | DC output (kW) for each individual plant at that bus |
| `sum_pp` | Total aggregated DC output (kW) across all plants at the bus |

**Geopanel** (`pvwatts_geopanel.csv`):

Wide-format file with one column per WECC bus, rows as hourly timestamps, values as total bus-level DC output in kW. This is the primary input for downstream node-level analysis.

### Full Pipeline Order

```
1. python pvwatts_pipeline.py
        ↓ writes pvwatts_powerplants/{bus}_{name}_{lat}_{lon}.csv

2. python agg-powerplant-predictions-node.py
        ↓ writes pvwatts_bus_agg/{bus}.csv
        ↓ writes pvwatts_geopanel.csv
```
