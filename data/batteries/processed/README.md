# `processed/` — file manifest

Outputs of the five notebooks in `../notebooks/`. Full method, results and the
data-quality log are in **`../README.md`**; this file is a manifest so that
someone arriving here can find the right table without reading the whole thing.

**Start here:** the four tables under *Model inputs* and *Validation* are what
the WECC model consumes. Everything else is supporting evidence for the choices
behind them, or an intermediate kept so the results can be audited to source.

---

## Model inputs — power, power limits, device configuration

| File | Grain | Rows / nodes | What it gives the model |
|---|---|---|---|
| **`node_reactive_bounds_by_year.csv`** | node × year | 160 / 49 | **The P–Q feasible region.** `max_charge_MW` and `max_discharge_MW` bound the real power variable; `reactive_MVAR` bounds the reactive variable; `max_P_MW` with `reactive_MVAR` sizes the disc; `ratio_aggregate` selects which half-plane cut applies. Use this for anything dispatch-related. |
| **`node_capacity_by_year.csv`** | node × year | 167 / 52 | **Fleet inventory.** Nameplate MW and MWh, max charge/discharge rates, plant count. Broader coverage than the bounds table (see *Coverage*) — use it for inventory totals, not for dispatch bounds. |
| **`technology_by_node_year.csv`** | node × year × technology | 172 / 50 | **Chemistry at each node.** One row per chemistry present: `nameplate_MW`, `energy_MWh`, `generators`, and the within-node shares `share_MW` / `share_MWh`. Each node-year's shares sum to 1. |

### Composing the P–Q region, per node-year

| `ratio_aggregate` | Region |
|---|---|
| ≈ 1 | The disc at that size, uncut |
| < 1 | Disc intersected with \|Q\| ≤ `reactive_MVAR` (horizontal band) |
| = 0 | Region degenerates — replace with a 1-D bound on P and pin Q at zero. **6 node-years** |
| > 1 | Disc intersected with \|P\| ≤ `max_P_MW` (vertical band) |

Two further shapes to expect: **17 node-years** have asymmetric charge and
discharge bounds, so the box on P is not symmetric about the origin; **29 of
160 node-years** lie above 1.1 and bind on real power rather than reactive.
Both half-planes must be composed from the per-node values — the fleet average
is not a rule that holds node by node.

---

## Validation — energy totals and observed shapes

| File | Grain | Rows / nodes | Use |
|---|---|---|---|
| **`node_generation_by_month.csv`** | node × year × month | 1,908 / 50 | **Reported charge, discharge and net generation.** The observed monthly shapes to compare simulated dispatch against, and the energy totals for true-up. |

Two conventions that matter when reading it:

- `NaN` ≠ 0. About 8% of node-months are `NaN`, meaning every plant at that
  node reported nothing to EIA-923 that month. "Did not report" and "genuinely
  zero" are kept distinct; the fill-or-skip decision is left to the consumer.
- 72.9% of node-months have negative net generation. This is correct —
  charging exceeds discharging by the losses.

**Weighting convention throughout: nameplate capacity is the modeling
parameter; reported energy is for true-up.**

---

## Coverage — read before joining these tables

The node tables do **not** cover the same node-years, by design. Joining them
naively will silently drop rows.

| Table | Node-years | Nodes |
|---|---|---|
| `node_capacity_by_year.csv` | 167 | 52 |
| `technology_by_node_year.csv` | 172 rows / 161 node-years | 50 |
| `technology_node_mixing.csv` | 161 | 50 |
| `node_reactive_bounds_by_year.csv` | 160 | 49 |
| `node_generation_by_month.csv` | 159 | 50 |

`node_capacity_by_year` is a superset of the others. The exact differences:

| Node-year | Present in | Absent from | Why |
|---|---|---|---|
| `9q9m24`, all five years | capacity | bounds, technology | Every generator at this node fails the `Status ∈ {OP, SB}` filter that the generator-grain pass applies. It holds capacity but nothing dispatchable. |
| `9x0mgn`, 2022 | capacity | bounds, technology, generation | Reported in EIA-860 (capacity) but never in EIA-923 (no operations ever filed). |
| `9q9pk4`, 2022 | capacity, technology | bounds | Removed by the reactive filing screen. |

**Nodes with no usable storage bound must be handled explicitly downstream —
not treated as zero by default.** The bounds and technology tables also cover
only nodes that host batteries; the rest of the WEC-240 nodes have no row at
all, and the convention for those still needs to be agreed.

### `nameplate_MW` differs between the capacity and bounds tables

`node_capacity_by_year.csv → Nameplate Capacity (MW)` and
`node_reactive_bounds_by_year.csv → nameplate_MW` are **not identical**. They
disagree on **15 of the 160 shared node-years**, capacity always the larger:

| Year | Capacity table | Bounds table |
|---|---|---|
| 2022 total | 5,239.8 MW | 5,231.1 MW |

The gap is the `Status ∈ {OP, SB}` filter, which the generator-grain pass
applies and the plant-grain pass does not. Neither figure is wrong — the
capacity table is the inventory, the bounds table is the dispatchable subset
with its operating envelope.

**For dispatch modelling use the bounds table.** For inventory totals use the
capacity table. Do not mix the two in one calculation.

### Column naming

The two passes use different conventions and this has not been reconciled: the
plant-grain tables carry the EIA header text verbatim (`Nameplate Capacity
(MW)`), the generator-grain tables use snake_case (`nameplate_MW`). Key columns
are `geohash` and `Year` in both.

### Time grain — open question

All configuration tables are **annual**, because EIA-860 is an annual filing.
The model wants a configuration for an arbitrary month. Two options:
forward-fill the annual value across twelve months, or use each generator's
operating month to place the step inside the year. The second is more faithful
— the fleet roughly doubled within some of these years — but needs the
operating-date field. **Not yet decided.**

---

## Supporting analysis

Evidence behind the modelling choices above. Not consumed by the model.

| File | Grain | Rows | Shows |
|---|---|---|---|
| `technology_node_mixing.csv` | node × year | 161 | Node summary rolled up from `technology_by_node_year.csv`: `technology_dominant`, `dominant_share_MW`, `mixed`. Use the breakdown, not this, when a node holds more than one chemistry. |
| `fleet_reactive_ratio_by_year.csv` | year | 5 | Five estimators of Q / max(P). They disagree, and the disagreement is the finding: every capacity-weighted estimator is below unity (pooled 0.672) while the median sits at exactly 1.000. |
| `node_ratio_weighting_by_year.csv` | year | 5 | Weighted against unweighted node means, and top-10 concentration. Low-ratio nodes carry 80–94% of the megawatts. |
| `reactive_screen_sensitivity.csv` | threshold | 7 | The headline figures move ~1 point across thresholds from 20 down to 3. Screening at all is load bearing; the threshold is not. |
| `technology_mix_by_year.csv` | year × technology | 20 | Fleet mix under three denominators (units, MW, MWh), which disagree substantially. |
| `applications_summary.csv` | flag | 11 | The eleven declared-application flags with response rates. |
| `applications_by_year.csv` | flag × year | 55 | The same, by year. |

> **The application flags cannot carry evidentiary weight.** No field is
> answered by even half the fleet (response rates 5.9%–49.1%, median 25.9%),
> and respondents answer `Y` 61–100% of the time. Non-response, not `N`, is how
> a negative is expressed. Read every proportion against its response rate.

---

## Base and intermediate tables

Kept so results can be audited back to source. Not deliverables.

| File | Grain | Rows | Notes |
|---|---|---|---|
| `battery_units_2018_2022.csv` | generator × year | 440 | Generator-grain base table, 36 columns. Everything in *Supporting analysis*, the bounds table and the technology tables derive from this. |
| `plant_to_node.csv` | plant | 149 | Which node each WECC plant matched to, and the match distance. **Any downstream work keyed on an older node assignment must be re-run against this file** — the nearest-node search changed and individual plants moved between nodes. |
| `battery_panel_2018_2022.csv` | plant × month | 12,733 | Full-US panel, EIA-860 + EIA-923 joined. |
| `battery_panel_labeled.csv` | plant × month | 8,251 | The WECC subset with the geohash label attached. |
| `plant_923_not_in_860.csv` | plant × year | 41 | Orphan log: present in EIA-923 but not EIA-860, so no coordinates and no node match. Excluded. |

---

## Figures

| File | Shows |
|---|---|
| `fig_node_map.png` | Nodes on the map; marker area = capacity, colour = ratio. The tabular form is `node_reactive_bounds_by_year.csv`. |
| `fig_ratio_by_year.png` | Five estimators of Q / max(P), by year |
| `fig_ratio_distribution.png` | Distribution of the ratio, per year |
| `fig_node_capacity_vs_ratio.png` | Node ratio against node size (log MW) |
| `fig_technology_mix.png` | Fleet mix under three denominators |
| `fig_lithium_share.png` | Dominant-chemistry share under each denominator |
| `fig_application_response.png` | Response rate against yes-share, per flag |
| `fig_reactive_application.png` | Reactive support flag, per year |

---

## Known issues

Summarised here; full log in `../README.md`.

1. **Nameplate Energy Capacity (MWh) looks high — OPEN.** 17,831.8 MWh for
   WECC in 2022 against EIA's published *national* figure of ~11,105 MWh.
   Likely some hybrid (solar + storage) plants reporting a whole-facility MWh.
   **Needs a decision before the MWh column is used.** The MW columns are
   unaffected.
2. **Half the fleet reports Q exactly equal to P.** 214 of 432 screened
   generator-years report a ratio of exactly 1.000 and **none at all** fall
   between 0.9 and 1.0 — the shoulder a measured quantity would populate. Treat
   the median as a filing artifact, not a measurement.
3. **One plant is screened out in all five years.** Casa Mesa Wind Energy
   Center Hybrid files 12.5 MVAR against a 1.0 MW battery. The site is a hybrid
   wind + battery facility (51 MW wind, 1 MW battery, 12.5 MVAR of power
   electronics) and the reactive rating is a facility-level figure, so it is
   not comparable with the battery's own real power rating. Flagged in
   `screened_out`.
4. **The mixed-node capacity share is reported pooled, not per year.** 11 of
   161 node-years hold more than one chemistry, pooled as 11.93% of capacity —
   but that sums a stock across five years and is dominated by 2022. Per year
   the share at mixed nodes is 5.7%, 9.6%, 50.4%, 13.7%, 7.2%. The capacity
   *not* held by each node's dominant chemistry is far smaller: under 0.65% in
   every year. See `../README.md`, step 3.
5. **Large match distances in the interior West.** 15 of 149 plants match more
   than 100 km from their nearest node (max 442.7 km, median 21.8 km). This is
   WEC-240 resolution, not a coordinate error. All retained;
   `match_dist_km` records the distance.

---

Scope throughout: **WECC only**, **2018–2022**, **EIA data only**.
Questions: `bennetm [at] nlr [dot] gov`.
