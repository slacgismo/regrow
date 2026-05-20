# REGROW Weather Forecast Pipeline

Parallel weather forecast data pipeline using Herbie and Dask to extract HRRR and GEFS model output at wind turbine site locations and store results to S3.

## Overview

This pipeline pulls historical NWP (Numerical Weather Prediction) forecast data for wind farm sites sourced from the US Wind Turbine Database (USWTDB). It runs two model pipelines in sequence for each monthly chunk across 2018–2022:

- **HRRR** — High-Resolution Rapid Refresh, hourly forecasts at 1–18 hour horizons
- **GEFS** — Global Ensemble Forecast System (member p01), 6-hourly forecasts at 24–72 hour horizons

Results are written as CSV files to S3 at `s3://pvdrdb-transfer/REGROW/herbie_forecasts/raw/`.

## Dependencies

```
herbie-data
dask[distributed]
dask-jobqueue
boto3
pandas
numpy
pvdrdb_tools       # internal NREL package
```

Install via conda:
```bash
conda install -c conda-forge herbie-data dask dask-jobqueue boto3 pandas numpy
```

## Configuration

The pipeline reads AWS credentials and database connection info from `pvdrdb_tools.PVDRDBQuery()`. Ensure your PVDRDB credentials are configured before running.

Wind turbine site locations are read from `uswtdb.csv`, which must be present in the working directory. Expected columns: `name`, `latitude`, `longitude`. Sites are averaged to one point per named location.

## Running on Kestrel (HPC)

The SLURM job script is located at `slurm/slurm.sh`. It requests a single node with 20 CPUs, 32GB RAM, and a 24-hour wall time limit under the `pvfleets24` account.

**Submit the job:**
```bash
cd /kfs2/projects/pvfleets24/repos/regrow/data
sbatch slurm/slurm.sh
```

**Monitor the job:**
```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f /kfs2/projects/pvfleets24/repos/regrow/data/slurm_outputs/herbie_<JOBID>.out

# Check for errors
tail -f /kfs2/projects/pvfleets24/repos/regrow/data/slurm_outputs/herbie_<JOBID>.err
```

**Cancel a running job:**
```bash
scancel <JOBID>
```

Email notifications are sent to `kirsten.perry@nrel.gov` on job start, completion, and failure. SLURM output and error logs are written to `/kfs2/projects/pvfleets24/repos/regrow/data/slurm_outputs/`.

## Running Manually

To run interactively on Kestrel or a local machine, activate the project conda environment first:

```bash
# Load mamba module (Kestrel only)
ml mamba

# Activate the project environment
mamba activate /kfs2/projects/pvfleets24/envs/regrow

# Navigate to the working directory
cd /kfs2/projects/pvfleets24/repos/regrow/data

# Run the pipeline
python generate_herbie_forecasts.py
```

For an interactive SLURM session instead of a batch job (useful for debugging):
```bash
salloc --account=pvfleets24 --partition=standard --nodes=1 --cpus-per-task=20 --mem=32G --time=2:00:00

# Once the session starts, activate the environment and run as above
ml mamba
mamba activate /kfs2/projects/pvfleets24/envs/regrow
cd /kfs2/projects/pvfleets24/repos/regrow/data
python generate_herbie_forecasts.py
```

## Usage

The script runs from `__main__` and processes data month by month from 2018-01-01 through 2022-12-31. Existing S3 files are checked at startup and skipped — the pipeline is safe to re-run after interruption.

## Output Format

Each task writes one CSV file per `(date, time_horizon)` pair:

**Filename:** `YYYY-MM-DD_HH_MM_SS_{N}hr.csv`

**Columns:**

| Column | Description |
|---|---|
| `longitude` | Site longitude |
| `latitude` | Site latitude |
| `forecast_time` | Forecast initialization time (UTC) |
| `time_horizon_hrs` | Forecast lead time in hours |
| `tag` | Variable name (e.g. `UGRD:80 m`) |
| `value` | Extracted forecast value |

HRRR files also include `wind_site_name`, `point_latitude`, `point_longitude`, and `forecast_horizon_hrs`.

## Variables Extracted

**HRRR (`prs` product):**

| Tag | Description |
|---|---|
| `TCDC:entire atmosphere` | Total cloud cover |
| `UGRD:80 m above ground` | U-component wind at 80m |
| `VGRD:80 m above ground` | V-component wind at 80m |
| `RH:2 m above ground` | Relative humidity |
| `PRES:surface` | Surface pressure |
| `TMP:surface` | Surface temperature |
| `DPT:2 m above ground` | Dewpoint temperature |

**GEFS (`atmos.5b` product, member p01):**

| Tag | Description |
|---|---|
| `UGRD:80 m` | U-component wind at 80m |
| `VGRD:80 m` | V-component wind at 80m |
| `TCDC` | Total cloud cover (475 mb) |
| `TMP:surface` | Surface temperature |
| `DPT:2 m` | Dewpoint temperature |

## Parallelism

Tasks are parallelized with `dask.compute(..., num_workers=20)`. Each function is decorated with `@delayed` and includes 3 retry attempts with 5-second backoff on failure.

The date range is chunked monthly to keep the Dask task graph manageable.

## Known Issues and Limitations

**GEFS archive availability:** GEFS `atmos.5b` for dates prior to ~2020 is not available on AWS or Google Cloud. The pipeline falls through to NOMADS, which redirects archived data to HPSS tape storage. These requests return HTTP 302 and fail silently. Dates in 2018–2019 will have a high failure rate for GEFS.

**GEFS init time alignment:** GEFS runs only at 00/06/12/18Z. If input dates include off-cycle hours, Herbie will return `None` for the file path. Use `snap_to_gefs_cycle()` before constructing the `Herbie` object.

**80m winds in 2018 GFS:** GFS did not output winds at 80m above ground until ~2019. If 80m is unavailable, extrapolate from 10m using the power law with `alpha=1/7`.

**HRRR coverage:** HRRR covers the CONUS domain only. Sites outside the continental US will silently return no data.

**SSL on NREL network:** NREL's network proxy intercepts HTTPS connections and inserts a self-signed certificate. This causes `SSLCertVerificationError` on requests to NCEI and NCAR RDA. Set `REQUESTS_CA_BUNDLE` to the NREL CA bundle or contact IT for the certificate.

## Logging

Logs are written to `example.log` (DEBUG level) and stdout (INFO level). Each task logs its date and time horizon on start, and logs failures including the exception message.

## S3 Structure

```
s3://pvdrdb-transfer/
  REGROW/
    herbie_forecasts/
      raw/
        2018-01-01_00_00_00_1hr.csv
        2018-01-01_00_00_00_2hr.csv
        ...
```
