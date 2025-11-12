# Input Files

* `config.glm`: The main configuration file which sets up the scenario to simulation.
* `wecc240.glm`: The main model file which should be run with command `gridlabd wecc240.glm`.
* `wecc240_psse.raw`: The original PSS/E model, which is an input to the `wecc240.glm` model.
* `controllers.py`: The control system model in Python.

# Output Files

* `wecc240_psse.glm`: The WECC system `glm` file converted from the PSS/E `raw` file.
* `wecc240.json`: The final state of the system after the simulation is complete.
* `wecc240_failed_*.csv`: The solver data of the last failed solution, if any.

# Data Flow

```mermaid
---
title: Data Flowa (Compilation)
---
graph LR

    HIFLD --> data/powerplants.csv.zip
    data/powerplants.csv.zip --> data/powerplants.py
    data/powerplants.py --> data/powerplants.glm

    wecc240_psse.raw --> gridlabd0[gridlabd]
    gridlabd0 --> wecc240_psse.glm
    wecc240_psse.glm --> gridlabd1[gridlabd]
    data/wecc240_gis.glm --> gridlabd1
    data/powerplants.glm --> gridlabd1
    data/powerplants_gis.glm --> gridlabd1
    gridlabd1 --> wecc240_raw.json
    wecc240_raw.json --> powerplants_aggregated.py
    powerplants_unknown.csv --> powerplants_aggregated.py
    gencosts.json --> powerplants_aggregated.py
    data/uspvdb.csv --> powerplants_aggregated.py
    data/uswtdb.csv --> powerplants_aggregated.py
    powerplants_aggregated.py --> powerplants_split.csv
    powerplants_aggregated.py --> powerplants_data.csv
    data/geodata/temperature.csv --> powerplants_aggregated.py
    powerplants_aggregated.py --> powerplants_aggregated.csv
    powerplants_aggregated.py --> powerplants_aggregated.glm
```

```mermaid
---
title: Data Flow (Simulation)
---
graph LR
  config.glm --> wecc240.glm
  wecc240_psse.raw --> wecc240.glm
  data/wecc240_gis.glm --> wecc240.glm
  powerplants_aggregated.glm --> wecc240.glm
  data/loads.glm --> wecc240.glm
  wecc240.glm --> wecc240.json
  wecc240.glm --> recorders/*
```
