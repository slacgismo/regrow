# Battery Data

This is the working directory for battery-related data and analysis in the
REGROW project.

## What goes here

Battery system data and analysis code relevant to the REGROW project: raw
datasets, processed/cleaned outputs, fixture files for testing, Python scripts,
and notebooks for data processing steps and numerical experiments. Examples:

* Battery capacity and specifications (by node, region, or technology class)
* Charge/discharge time series
* State-of-charge profiles
* Cost and degradation parameters
* Scripts that fetch, clean, or transform data
* Notebooks that run experiments or produce results figures

## Directory structure

```
data/batteries/
├── README.md          ← you are here
├── raw/               ← data as downloaded, unmodified (EIA-860 / EIA-923)
├── processed/         ← cleaned or transformed outputs
└── notebooks/         ← marimo notebooks for the processing steps
```

Add subdirectories as the work warrants (e.g. a `fixtures/` folder for small,
stable test inputs). Feel free to add, rename, or reorganize — just keep this
README up to date so the next person can orient quickly.

## Adding data files to git

Data files under the [GitHub 100 MB file size limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
can be committed directly and pushed to GitHub — no special tooling needed. For
files above that limit, coordinate with the team (options include Git LFS or
storing outside the repo and documenting the source here).

## Questions

Reach out to `bennetm [at] nlr [dot] gov` with any questions about the project
or this dataset area.

---

# Current contents: EIA battery storage → WEC-240 node aggregation (2018–2022)

Maps EIA grid-scale battery storage data onto the WEC-240 network model: every
battery plant is labeled with its nearest model node by great-circle distance,
then capacity and operations are aggregated **by node, not by generator**.

Scope: **WECC only**, **2018–2022**, **EIA data only** (no CAISO in this pass).

## Files

**`processed/`**

| File | Grain | Description |
|---|---|---|
| `node_capacity_by_year.csv` | node × year | **Deliverable.** MW, MWh, charge/discharge rates, plant count |
| `node_generation_by_month.csv` | node × year × month | **Deliverable.** Charge, discharge, net gen (long format) |
| `plant_to_node.csv` | plant | Which node each WECC plant matched to, and the distance |
| `plant_923_not_in_860.csv` | plant × year | Orphan log — see edge case 3 below |
| `battery_panel_2018_2022.csv` | plant × month | Intermediate: full-US panel, EIA-860 + EIA-923 joined |
| `battery_panel_labeled.csv` | plant × month | Intermediate: WECC panel with geohash label attached |

**`notebooks/`**

| File | Does |
|---|---|
| `build_battery_panel.py` | Step 1 — EIA-860 + EIA-923 → monthly plant panel |
| `nearest_node.py` | Steps 2–3 — WECC filter, nearest-node match, aggregation by node |

Both are [marimo](https://marimo.io) notebooks: each variable is defined in
exactly one cell, and multi-step logic lives inside functions.

```bash
cd data/batteries/notebooks
marimo edit build_battery_panel.py   # then: marimo edit nearest_node.py
```

Paths inside the notebooks are resolved relative to the notebook's own
location, so they run from a fresh clone with no editing.

## Dependencies elsewhere in the repo

- `utils.py` — supplies `geohash()`, `nearest()`, `haversine_distance()`
- `nodes.csv` — unique WEC-240 node locations, with `geocode` + `Lat` / `Long`

Neither is duplicated here; both are read by relative path from their existing
location in the repo.

One environment note: `utils.py` reads `HOME` at import time and imports `psm3`
at the top. On Windows neither is available, so cell 1 of `nearest_node.py`
sets a `HOME` fallback and stubs out `psm3`. This is an environment shim only —
it changes no calculation.

## Raw data

`raw/` holds the EIA workbooks as downloaded, unmodified:

```
raw/EIA860/eia860{YEAR}/2___Plant_Y{YEAR}.xlsx
raw/EIA860/eia860{YEAR}/3_4_Energy_Storage_Y{YEAR}.xlsx
raw/EIA923/EIA923_Schedules_2_3_4_5_M_12_{YEAR}_Final_Revision.xlsx
```

Years used: 2018, 2019, 2020, 2021, 2022.

- EIA-860: https://www.eia.gov/electricity/data/eia860/
- EIA-923: https://www.eia.gov/electricity/data/eia923/

## Method

**Step 1 — monthly panel (EIA-860 × EIA-923).**
EIA-860 gives the inventory: which plants hold batteries, where they are
(lat/long), how big they are. The filter is `Prime Mover == "BA"`, which keeps
batteries and excludes flywheels (FW), compressed air (CP), and so on. A plant
may hold several battery generators, so capacity is summed to plant level and a
`Generator Count` is recorded.

EIA-923 gives the operations. EIA treats electricity as the *fuel* for storage,
so the column semantics are:

| EIA-923 column | Meaning here |
|---|---|
| `Quantity {month}` | gross **charge** (MWh) |
| `Grossgen {month}` | gross **discharge** (MWh) |
| `Netgen {month}` | discharge − charge (usually negative) |

The 12 wide monthly columns are reshaped to long — one row per plant-month. The
join uses EIA-860 as the left anchor, because only 860 carries coordinates, and
coordinates are what the node match needs.

**Step 2 — nearest-node match.**
Many WEC-240 graph nodes are co-located at the same site, so the 243 graph nodes
collapse to ~126 distinct physical locations in `nodes.csv` (where the geohash
column is named `geocode`). The match uses the repo's own `utils.geohash()` to
encode each plant's lat/long and `utils.nearest()` to find the closest node
hash, keeping the whole chain on the repo's geohash convention rather than
introducing a second distance implementation. `utils.nearest()` returns
**meters**; the notebook converts to km.

The WECC filter is applied *first*, before matching. This is the electrically
correct footprint: a pure distance cut would wrongly admit ERCOT (Texas) and MRO
(Midwest) plants sitting near the western edge, and would force Hawaii onto a
California node. `match_dist_km` is kept as a data-quality flag, not a filter —
nothing is dropped on distance.

**Step 3 — aggregate by node.**
Capacity is a per-plant-*year* attribute, constant month to month, so the code
takes `.first()` per (node, year, plant) *before* summing across the plants at
that node — otherwise the 12 monthly rows would multiply capacity by 12.

Generation uses `sum(min_count=1)` so a node-month where *every* plant was
missing from EIA-923 stays `NaN` rather than collapsing to `0`. "Did not report"
and "genuinely zero" are different facts.

## Results

**Capacity table** — 173 rows, 53 nodes.

| Year | Nameplate MW |
|---|---|
| 2018 | ~862 |
| 2019 | ~1,024 |
| 2020 | ~1,530 |
| 2021 | ~4,772 |
| 2022 | **5,239.8** |

The 2021 jump matches EIA's published narrative for WECC storage build-out. The
2022 total reconciles exactly with the pre-aggregation WECC plant total, which
confirms nothing is lost or double-counted in the groupby.

**Generation table** — 1,956 rows, 51 nodes. Round-trip efficiency
(discharge ÷ charge) rises 79% → 86% across the five years. About 73% of
node-months have negative net generation, which is physically correct: charging
exceeds discharging by the round-trip loss.

## Data-quality log

**1. Capacity table has 53 nodes, generation table has 51.**
Two nodes host batteries that appear in EIA-860 (capacity reported) but never in
EIA-923 (no operations ever reported) — e.g. plants commissioned late in a year.
Not a bug; a property of the source data.

**2. `NaN` rows in the generation table (~8%).**
Same cause at the node-month level: every plant at that node reported nothing
that month. Deliberately preserved as `NaN` (see `min_count=1` above).
Downstream users need to decide whether to fill with 0 or skip — the choice is
intentionally left open here.

**3. Orphan plants — present in EIA-923 but not EIA-860.**
Logged per year in `processed/plant_923_not_in_860.csv`. These have operations
data but no coordinates, so they cannot be matched to a node and are excluded.

**4. Nameplate Energy Capacity (MWh) looks high — OPEN QUESTION.**
This table reports ~17,832 MWh for WECC in 2022, while EIA's published
*national* figure for end-2022 is ~11,105 MWh. Most likely cause: some hybrid
(solar + storage) plants report a whole-facility MWh that includes the PV side
rather than the battery alone. This needs a decision before the MWh column is
used. The MW columns are unaffected.

**5. Large match distances in the interior West.**
A handful of AZ/NM/CO plants match 100–450 km from their nearest node. This is
WEC-240 model resolution — sparse nodes in the interior West — not a coordinate
error. All are retained, and `match_dist_km` records the distance for each.

**6. MW ≠ charge rate ≠ discharge rate for ~27% of rows.**
Expected. Nameplate capacity, max charge rate, and max discharge rate are
usually equal but not necessarily so; the data preserves the difference rather
than assuming it away.
