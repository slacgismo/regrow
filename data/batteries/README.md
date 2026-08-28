# Battery Data

Battery system data and analysis for the REGROW project: raw datasets,
processed outputs, and the notebooks that produce them.

```
data/batteries/
├── README.md          ← you are here
├── raw/               ← EIA-860 / EIA-923 workbooks as downloaded, plus lookups
├── processed/         ← every table and figure lands here; see processed/README.md
└── notebooks/         ← marimo notebooks
```

Data files under the [GitHub 100 MB limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
can be committed directly. For anything larger, coordinate with the team.

Questions: `bennetm [at] nlr [dot] gov`.

---

# EIA battery storage for the WEC-240 model, 2018–2022

Scope throughout: **WECC only**, **2018–2022**, **EIA data only**.

Every WECC battery plant is labelled with its nearest WEC-240 node, then
capacity, operations and per-generator attributes are aggregated by node. The
result is what the dispatch model needs to place and configure storage at each
node, and what its output is checked against.

## What the model consumes

**Configuration — placement, sizing, feasible region**

| File | Grain | Rows / nodes | Role |
|---|---|---|---|
| `node_reactive_bounds_by_year.csv` | node × year | 160 / 49 | The P–Q feasible region per node |
| `node_capacity_by_year.csv` | node × year | 167 / 52 | Inventory: nameplate MW and MWh, charge/discharge rates |
| `technology_by_node_year.csv` | node × year × technology | 172 / 50 | Which chemistry sits at each node, and how much |

**Validation — energy totals and observed shapes**

| File | Grain | Rows / nodes | Role |
|---|---|---|---|
| `node_generation_by_month.csv` | node × year × month | 1,908 / 50 | Reported charge, discharge and net generation |

Everything else in `processed/` is supporting evidence or an audit trail;
`processed/README.md` is the per-file manifest.

**Nameplate capacity is the modeling parameter; reported energy is for
true-up.** All configuration tables are annual, because EIA-860 is an annual
filing — see open question 2.

## Pipeline

```
EIA-860 + EIA-923 ──► build_battery_panel.py ──► battery_panel_2018_2022.csv
                                                          │
                                                   nearest_node.py
                                                          │
                    ┌─────────────────────────────────────┼──────────────────────────┐
                    ▼                                     ▼                          ▼
             plant_to_node.csv              node_capacity_by_year.csv    node_generation_by_month.csv
                    │
                    │ node labels only
                    ▼
EIA-860 ──────► build_unit_table.py ──► battery_units_2018_2022.csv
                                        (generator-year grain)
                                                   │
                          ┌────────────────────────┴────────────────────────┐
                          ▼                                                 ▼
              reactive_power_ratio.py                         tech_and_applications.py
        node_reactive_bounds_by_year.csv                  technology_by_node_year.csv
```

Pass 2 borrows only the node label from pass 1. Every capacity figure in pass 2
is rebuilt from EIA-860 at generator grain, because the pass-1 panel aggregates
with `groupby`–`sum`, which is correct for capacity and destroys the
per-generator attributes pass 2 needs.

Run in order — each step reads what the previous one writes:

```bash
cd data/batteries/notebooks
marimo edit build_battery_panel.py     # EIA-860 + EIA-923 → monthly plant panel
marimo edit nearest_node.py            # WECC filter, node match, aggregation
marimo edit build_unit_table.py        # EIA-860 → generator-year unit table
marimo edit reactive_power_ratio.py    # Q/P ratio; per-node P–Q bounds
marimo edit tech_and_applications.py   # per-node chemistry; declared applications
```

These are [marimo](https://marimo.io) notebooks: each variable is defined in
exactly one cell, cell-local names are underscore-prefixed, and every path
resolves from the notebook's own location, so they run from a fresh clone
without editing.

`raw/utils.py` supplies `geohash()`, `nearest2()` and `haversine_distance()`;
`raw/nodes.csv` holds the unique WEC-240 node locations. `utils.py` reads `HOME`
at import and imports `psm3` at the top; on Windows neither exists, so
`nearest_node.py` sets a `HOME` fallback and stubs `psm3`. Environment shim
only — it changes no calculation.

---

# Pass 1 — plant to node

## Method

**Monthly panel.** EIA-860 gives the inventory: which plants hold batteries,
where they are, how big they are. The filter is `Prime Mover == "BA"`, which
keeps batteries and excludes flywheels, compressed air and the rest. EIA-923
gives the operations, and treats electricity as the *fuel* for storage, so
`Quantity` is gross charge, `Grossgen` is gross discharge, and `Netgen` is
discharge − charge (usually negative). The 12 wide monthly columns are reshaped
to long. The join uses EIA-860 as the left anchor, because only 860 carries
coordinates and coordinates are what the node match needs.

**Nearest-node match.** The 243 WEC-240 graph nodes collapse to ~126 distinct
physical locations, many being co-located. The match uses the repo's own
`utils.geohash()` and `utils.nearest2()` rather than a second distance
implementation. The **WECC filter is applied first**, before matching: a pure
distance cut would wrongly admit ERCOT and MRO plants near the western edge and
force Hawaii onto a California node. `match_dist_km` is a data-quality flag, not
a filter — nothing is dropped on distance.

**Aggregation.** Capacity is a per-plant-*year* attribute, so the code takes
`.first()` per (node, year, plant) before summing across plants at a node —
otherwise the 12 monthly rows would multiply capacity by 12. Generation uses
`sum(min_count=1)`, so a node-month where every plant was missing from EIA-923
stays `NaN` rather than collapsing to `0`.

## Results

149 WECC battery plants match onto 52 node locations. Match distance: median
21.8 km, mean 46.1 km, max 442.7 km.

| Year | Nameplate MW | Nameplate MWh | discharge ÷ charge |
|---|---|---|---|
| 2018 | 285.6 | 630.1 | 78.9% |
| 2019 | 320.2 | 787.6 | 82.5% |
| 2020 | 598.3 | 1,059.6 | 84.8% |
| 2021 | 2,584.1 | 7,127.3 | 86.8% |
| 2022 | **5,239.8** | **17,831.8** | 86.1% |

The fleet grows roughly 18× in power over five years, with the step change in
2021, matching EIA's published narrative for WECC storage build-out. The 2022
total reconciles exactly with the pre-aggregation WECC plant total.

72.9% of node-months have negative net generation, which is physically correct:
charging exceeds discharging by the losses.

---

# Pass 2 — generator grain

Storage technology, enclosure type, the nameplate reactive power rating and the
eleven application flags are all **generator-level** facts, and several plants
hold more than one battery generator. Aggregating to the plant first destroys
them, so pass 2 rebuilds from EIA-860 at generator grain and aggregates to the
node once, downstream, where it is auditable.

## The unit table

One row per (Plant Code, Generator ID, Year): **440 generator-years, 146 plants,
50 nodes**, 36 columns. EIA header text moves between vintages, so columns are
matched on a normalised key plus a prefix and every year prints its row count
and any column it failed to find — a renamed field cannot pass unnoticed as
silent nulls.

Three scope decisions:

| Decision | Rationale |
|---|---|
| `Status ∈ {OP, SB}` | EIA-860 Instructions Table 4 records **availability**, not presence. SB is available but not normally used, and is dispatchable; OS and OA are out of service. Summing the latter into a nodal bound lets the model schedule capacity that cannot respond. |
| Reactive rating read as **MVAR**, not MVA | Instructions line 38 specifies MVAR. Under an MVA reading, S ≥ P holds identically and a rating above the real power rating carries no information; under MVAR it is a statement about converter hardware. This is what makes the Q > P cases substantive rather than arithmetic. |
| Blank flag ≠ `N` | The application flags are voluntary. A blank means the operator did not answer, not that the answer is no. |

## The P–Q feasible region

The conventional storage region is the converter thermal limit P² + Q² ≤ S² — a
disc. Heating depends on total current and is indifferent to phase angle, which
is what makes the disc the right primitive. But the disc reaches S on both axes,
so adopting it asserts that every unit's reactive rating equals its real power
rating. That assertion is testable:

```
r = Q_nameplate / max(P_charge, P_discharge)
```

Zero is data: a reported 0 MVAR declares no reactive capability and is retained
with ratio 0; only absent values are excluded. Records with r > 10 are screened
as units or decimal errors, and the notebook prints every excluded record in
full. Accounting: 440 read → 437 with a defined ratio → **432 after the screen**.

### The uncut disc is inadequate in every year

![Q over max P by year](processed/fig_ratio_by_year.png)

| Year | Generators | Energy-wtd | Power-wtd | Unweighted | Median | Aggregate ΣQ/ΣP |
|---|---|---|---|---|---|---|
| 2018 | 46 | 0.766 | 0.780 | 0.958 | 1.000 | 0.778 |
| 2019 | 60 | 0.894 | 0.841 | 1.052 | 1.000 | 0.840 |
| 2020 | 70 | 0.690 | 0.483 | 1.025 | 1.000 | 0.481 |
| 2021 | 106 | 0.922 | 0.835 | 0.928 | 1.000 | 0.836 |
| 2022 | 150 | 0.578 | 0.594 | 0.838 | 1.000 | 0.598 |
| **pooled** | **432** | **0.685** | **0.670** | 0.933 | 1.000 | **0.672** |

Every year lies below unity on every capacity-weighted estimator: the horizontal
restriction |Q| ≤ Q_max is required, not optional. Five estimators are reported
because they disagree, and the disagreement is the finding. The aggregate is
what a nodal model realises — one node, one reactive budget, one power budget.

### The median of 1.000 is a filing artifact

![Distribution of Q over max P](processed/fig_ratio_distribution.png)

53.2% of screened generator-years lie within ten percent of unity, but 49.5%
report **exactly 1.000** and **zero records** sit strictly between 0.9 and 1.0 —
the shoulder a measured quantity would populate. A reactive rating equal to the
real power rating to the last digit is more consistent with the field being
copied across than with converter capability. Quote the capacity-weighted
estimators, which are driven by the large units that do report distinct values.

### Low-ratio nodes carry the capacity

![Node ratio against node size](processed/fig_node_capacity_vs_ratio.png)

| Year | Nodes | Unweighted | Capacity-weighted | Top-10 share of MW | Top-10 weighted |
|---|---|---|---|---|---|
| 2018 | 22 | 0.922 | 0.780 | 93% | 0.770 |
| 2019 | 22 | 0.963 | 0.842 | 92% | 0.828 |
| 2020 | 28 | 0.910 | 0.483 | 94% | 0.452 |
| 2021 | 40 | 0.866 | 0.835 | 81% | 0.792 |
| 2022 | 48 | 0.805 | 0.598 | 80% | 0.547 |

The capacity-weighted mean is below the unweighted mean in every year, and the
ten largest nodes — holding 80–94% of the megawatts — sit lower still. An
unweighted read overstates reactive headroom precisely where the storage is.

### The orientation is not uniform across nodes

![Nodes on the map](processed/fig_node_map.png)

Marker area is installed capacity, colour is the node ratio. The large markers
that appeared in 2021–2022 are predominantly at the low end of the scale.

But the fleet total is not a rule that holds node by node. **29 of 160
node-years lie above 1.1** and bind on real power rather than reactive; **17**
have asymmetric charge and discharge bounds, so the box on P is not symmetric
about the origin; and **6** have zero reactive capability, where the region
degenerates to a segment of the real power axis and no scaling of a disc
reproduces it. Both half-planes must be composed from the per-node bounds, not
hard-coded from the fleet average.

### Composing the region

Per node and year, from `node_reactive_bounds_by_year.csv`:

| `ratio_aggregate` | Region |
|---|---|
| ≈ 1 | The disc at that size, uncut |
| < 1 | Disc intersected with \|Q\| ≤ `reactive_MVAR` (horizontal band) |
| = 0 | Replace with a 1-D bound on P and pin Q at zero |
| > 1 | Disc intersected with \|P\| ≤ `max_P_MW` (vertical band) |

`max_charge_MW` / `max_discharge_MW` bound real power, `reactive_MVAR` bounds
reactive, `max_P_MW` with `reactive_MVAR` sizes the disc, `nameplate_MW` is the
weight for any node-level average, and `screened_out` flags records removed by
the filing screen.

### The screen matters; its threshold does not

| Threshold on r | Generators | Dropped | Energy-wtd | Aggregate | Unweighted |
|---|---|---|---|---|---|
| none / 20 | 437 | 0 | 0.692 | 0.679 | 1.065 |
| **10 (adopted)** | **432** | **5** | **0.685** | **0.672** | **0.933** |
| 5 | 432 | 5 | 0.685 | 0.672 | 0.933 |
| 3 | 431 | 6 | 0.679 | 0.665 | 0.926 |
| 2 | 413 | 24 | 0.632 | 0.628 | 0.850 |

Across thresholds from 20 down to 3 the pooled aggregate moves about one point.
The decision to screen at all is load bearing; the choice of threshold is not.

## Technology

The model configures one aggregate battery per node, so it needs the node's
chemistry, not the fleet's. Two tables answer that, the second rolled up from
the first so they cannot disagree.

**`technology_by_node_year.csv`** — one row per (Year, geohash, technology):
`generators`, `nameplate_MW`, `energy_MWh`, and the within-node shares
`share_MW` / `share_MWh`, which sum to 1 across each node-year. Long rather than
wide, because the set of technologies present changes between years.

**`technology_node_mixing.csv`** — one row per node-year: `technology_dominant`
(the chemistry holding the most nameplate MW), `dominant_share_MW`, and `mixed`.
Capacity is the weight rather than generator count, because the model acts on
capacity: a node can hold one large unit of one chemistry beside several small
units of another. `dominant_share_MW` is what says whether the label can be used
on its own — a node at 0.99 and a node at 0.51 are different facts.

### The fleet is lithium-ion, and increasingly so

![Technology mix under three denominators](processed/fig_technology_mix.png)

![Dominant chemistry share by denominator](processed/fig_lithium_share.png)

Four codes appear: **LIB** lithium-ion (417 generator-years), **NAB** sodium
based (12), **PBB** lead acid (6), **FLB** flow (5). Every generator-year
declares exactly one, and 100% of generator-years, MW and MWh carry a declared
chemistry, so the mix is a partition of the whole fleet.

| Year | Generators | by count | by power | by energy |
|---|---|---|---|---|
| 2018 | 40 of 47 | 85.11% | 95.87% | 89.26% |
| 2019 | 56 of 61 | 91.80% | 96.69% | 92.41% |
| 2020 | 68 of 73 | 93.15% | 98.36% | 94.50% |
| 2021 | 104 of 107 | 97.20% | 99.85% | 99.77% |
| 2022 | 149 of 152 | **98.03%** | **99.93%** | **99.91%** |

The three denominators disagree by enough to matter. In 2022 the non-lithium
remainder is 1.97% of generators but only 0.07% of power and 0.09% of energy —
three generators totalling 3.8 MW. The count column overstates the remainder by
a factor of about 25, which is why the lower row of the mix figure rescales to
the remainder: at these shares it is a few pixels tall at full scale.

### One chemistry describes almost every node

161 node-years across 50 nodes. Dominant chemistry: lithium-ion at 147,
sodium based at 7, lead acid at 5, flow at 2.

11 node-years hold more than one chemistry. Their pooled capacity is 11.93% of
the fleet, but that figure is misleading twice over: it sums a stock across five
years, and `mixed` counts the whole node — at `9muccv` that means 252 MW is
counted as mixed when 250 MW of it is lithium-ion. The operative quantity is the
capacity **not** held by the dominant chemistry:

| Year | Fleet MW | At mixed nodes | Not at dominant chemistry |
|---|---|---|---|
| 2018 | 281.2 | 5.7% | 1.8 MW (0.64%) |
| 2019 | 295.8 | 9.6% | 1.5 MW (0.51%) |
| 2020 | 596.3 | 50.4% | 3.5 MW (0.59%) |
| 2021 | 2,569.6 | 13.7% | 3.0 MW (0.12%) |
| 2022 | 5,232.8 | 7.2% | 3.0 MW (0.06%) |

`technology_dominant` therefore describes essentially the whole fleet — the
mislabelled capacity is under 0.65% in every year. Only two node-years are
genuinely ambiguous: `9we1bp` 2018 (dominant share 0.5556) and `9q97v8`
2019–2020 (0.8889). Both are small. Note also that half the 2020 fleet sat at
chemically mixed nodes, which the pooled figure hides.

Enclosure type is recorded but is not a model parameter. `CS` (containerized
stationary) is the largest category — 3,807 of 5,233 MW in 2022.

## Declared applications

Eleven voluntary Y/N flags. *Voltage or Reactive Power Support* is the only
field stating whether an installation is intended to provide voltage support at
all, and is therefore the one independent check on four-quadrant operation.

![Response rate against yes-share](processed/fig_application_response.png)

![Reactive support flag by year](processed/fig_reactive_application.png)

| Application | Y | N | blank | response rate | yes-share of respondents |
|---|---|---|---|---|---|
| Arbitrage | 185 | 31 | 224 | 49.1% | 85.6% |
| System Peak Shaving | 199 | 10 | 231 | 47.5% | 95.2% |
| Frequency Regulation | 163 | 3 | 274 | 37.7% | 98.2% |
| Excess Wind and Solar Generation | 140 | 21 | 279 | 36.6% | 87.0% |
| Load Management | 130 | 11 | 299 | 32.0% | 92.2% |
| Ramping / Spinning Reserve | 114 | 0 | 326 | 25.9% | 100.0% |
| Backup Power | 58 | 19 | 363 | 17.5% | 75.3% |
| Voltage or Reactive Power Support | 58 | 9 | 373 | 15.2% | 86.6% |
| Load Following | 56 | 8 | 376 | 14.5% | 87.5% |
| Co-Located Renewable Firming | 37 | 18 | 385 | 12.5% | 67.3% |
| Transmission and Distribution Deferral | 16 | 10 | 414 | 5.9% | 61.5% |

**These flags cannot carry evidentiary weight.** No field is answered by even
half the fleet, and where operators answer at all they overwhelmingly answer `Y`
— Ramping / Spinning Reserve records 114 `Y` against zero `N`. Non-response, not
`N`, is how a negative is expressed, so every proportion must be read against
its response rate.

In particular, the reactive support flag cannot settle four-quadrant operation:
it is answered on 67 of 440 generator-years, and the 86.6% yes-share among those
respondents is only 13.2% of the fleet. The reactive **rating** is reported on
437 of 440. The ratings carry the burden of evidence on capability; the flag can
corroborate but cannot establish it. The two fields are populated independently
and disagree on 9 generator-years.

---

# Coverage and data quality

## The node tables do not cover the same node-years

| Table | Node-years | Nodes |
|---|---|---|
| `node_capacity_by_year.csv` | 167 | 52 |
| `technology_by_node_year.csv` | 172 rows / 161 node-years | 50 |
| `node_reactive_bounds_by_year.csv` | 160 | 49 |
| `node_generation_by_month.csv` | 159 | 50 |

`node_capacity_by_year` is a superset. The differences:

| Node-year | Absent from | Why |
|---|---|---|
| `9q9m24`, all years | bounds, technology | Every generator fails the `Status ∈ {OP, SB}` filter. The node holds capacity but nothing dispatchable. |
| `9x0mgn`, 2022 | bounds, technology, generation | In EIA-860 but never in EIA-923 — no operations ever filed. |
| `9q9pk4`, 2022 | bounds | Removed by the reactive filing screen. |

**Nodes with no usable storage bound must be handled explicitly downstream, not
treated as zero by default.** The bounds and technology tables also cover only
nodes that host batteries — see open question 3.

## `nameplate_MW` differs between the capacity and bounds tables

They disagree on 15 of the 160 shared node-years, capacity always the larger:
5,239.8 MW against 5,231.1 MW in 2022. The gap is the `Status ∈ {OP, SB}`
filter, which the generator-grain pass applies and the plant-grain pass does
not. **Use the bounds table for dispatch and the capacity table for inventory
totals; do not mix them in one calculation.**

The two passes also use different column naming — plant-grain tables carry the
EIA header text verbatim (`Nameplate Capacity (MW)`), generator-grain tables use
snake_case (`nameplate_MW`). Key columns are `geohash` and `Year` in both.

## Known issues

1. **Nameplate Energy Capacity (MWh) looks high.** 17,831.8 MWh for WECC in 2022
   against EIA's published *national* figure of ~11,105 MWh. Most likely some
   hybrid solar + storage plants report a whole-facility MWh. The MW columns are
   unaffected, but energy-weighted figures inherit this.
2. **`NaN` in the generation table (~8% of node-months)** means every plant at
   that node reported nothing that month. Preserved rather than zero-filled:
   "did not report" and "genuinely zero" are different facts, and the fill-or-skip
   decision is left to the consumer.
3. **41 plant-years appear in EIA-923 but not EIA-860**, logged in
   `plant_923_not_in_860.csv`. No coordinates, so no node match; excluded.
4. **15 of 149 plants match more than 100 km from their nearest node**, up to
   442.7 km. This is WEC-240 resolution in the interior West, not a coordinate
   error. All retained; `match_dist_km` records the distance.
5. **One plant is screened out in all five years.** Casa Mesa Wind Energy Center
   Hybrid files 12.5 MVAR against a 1.0 MW battery. The site is a hybrid wind +
   battery facility (51 MW wind, 1 MW battery, 12.5 MVAR of power electronics),
   so the reactive rating is a facility-level figure and is not comparable with
   the battery's own real power rating.
6. **The unit table has 50 nodes, the bounds table 49.** One node contributes no
   generator with a defined ratio. Retained for documentation rather than
   silently reconciled.
7. **Node assignments moved when the nearest-node search changed.**
   `nearest_node.py` now uses `utils.nearest2()`, which is haversine throughout;
   the previous `utils.nearest()` ranked by squared degree difference between
   decoded geohash centres, which is not the same ordering away from the equator.
   Fleet totals are identical, but individual plants moved between nodes. **Any
   downstream work keyed on an older node assignment must be re-run against the
   current `plant_to_node.csv`.**
8. **MW ≠ charge rate ≠ discharge rate for ~27% of rows.** Expected; the data
   preserves the difference rather than assuming it away.

# Open questions

1. **Nameplate Energy Capacity (MWh)** — needs a decision before the MWh column
   is used. See known issue 1.
2. **Time grain.** The configuration tables are annual because EIA-860 is an
   annual filing, but the model wants a configuration for an arbitrary month.
   Forward-fill the annual value across twelve months, or use each generator's
   operating month to place the step inside the year? The second is more
   faithful — the fleet roughly doubled within some of these years — but
   requires the operating-date field. **Needs a decision before the tables are
   consumed.**
3. **Nodes with no storage.** Downstream code needs an agreed convention:
   absent rows, or explicit zero-capacity rows.
4. **Coverage beyond 2022.** The 2018 and 2019 figures rest on small samples,
   and later vintages dominate any pooled figure.
