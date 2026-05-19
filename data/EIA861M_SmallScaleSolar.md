# EIA-861M Small Scale Solar (Distributed Generation) Data

## Overview

This dataset provides monthly distributed solar generation and capacity estimates for WECC-service area states, sourced from the UIA (Utility Integration Analysis) published by EIA under the EIA-861M reporting framework.

## Source

EIA Electricity: Monthly Generator Profile -> Small Scale PV estimate section:
https://www.eia.gov/electricity/data/eia861m/

Each year is published as a standalone `.xlsx` file. There is no single bulk download.

## Files

| File | Path | Description |
|------|------|-------------|
| `small_scale_solar_2018to2022.xlsx` | `wecc240/` | Concatenation of the five yearly EIA-861M Small Scale PV xlsx downloads (2018-2022). Contains all states. |
| `wecc_dg_solar.csv` | (this directory) | Filtered copy: only WECC-service area states, exported from a filtered sheet tab in the xlsx above. |
| `WECC_DG_solar_analysis.py` | (this directory) | Python script that reads `wecc_dg_solar.csv` and produces the final bus-level output. |
| `wecc_bus_dg_cap_and_gen_by_month.csv` | (this directory) | Final output: distributed solar capacity and generation mapped to WECC buses, by month. |

## Reproduction Steps (Manual)

This pipeline is **not fully automated**. The following steps must be performed manually:

1. **Download yearly xlsx files** from the "Small scale PV estimate" section at https://www.eia.gov/electricity/data/eia861m/ for each year (currently 2018-2022).
2. **Concatenate** the downloaded xlsx files into a single workbook: `wecc240/small_scale_solar_2018to2022.xlsx`.
3. **Create a filtered sheet** within that workbook for only the states of interest (WECC service area).
4. **Export that sheet** as CSV to `wecc_dg_solar.csv` in this directory.
5. **Run** `WECC_DG_solar_analysis.py` to produce `wecc_bus_dg_cap_and_gen_by_month.csv`.

## Adding New Years

When EIA publishes a new year (e.g., 2023):

1. Download the new year's xlsx from the EIA URL above.
2. Append it to `wecc240/small_scale_solar_2018to2022.xlsx` (and rename accordingly).
3. Refresh the WECC-only filtered sheet and re-export `wecc_dg_solar.csv`.
4. Re-run `WECC_DG_solar_analysis.py` to regenerate the final output.

## Notes for Agents

- Steps 1-4 are manual and cannot be fully automated: the EIA site serves each year as a separate xlsx with no API or bulk download option. Scraping may be possible but the site requires manual navigation.
- Step 5 is scriptable and can be re-run independently when the CSV input is updated.
- The intermediate xlsx (`small_scale_solar_2018to2022.xlsx`) contains all states; the CSV (`wecc_dg_solar.csv`) is the WECC-filtered subset.
