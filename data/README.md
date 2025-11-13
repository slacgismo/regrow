To update the GLM file, run the Makefile.

    make

# Data Flows

## Temperature Data

```mermaid
---
title: Temperature Data Flow
---
graph TD

    NREL_amy2018 --> data/geodata.py
    data/geodata.py --> data/geodata/temperature_2018.csv

    NREL_amy2018 --> data/weather.py
    data/weather.py --> data/weather/temperature.csv
    data/weather/temperature.csv --> data/geodata_project_years.py
    
    data/geodata/temperature_2018.csv --> data/geodata_project_years.py
    data/geodata_project_years.py --> data/geodata/temperature.csv
    data/geodata/temperature.csv --> data/sensitivity.py
	data/sensitivity.py --> data/sensitivity.csv
	data/geodata/total.csv --> data/sensitivity.py
	data/sensitivity.py --> data/geodata/baseload.csv
```

## Solar Data

```mermaid
---
title: Solar Data Flow
---
graph TD

    NREL_amy2018 --> data/geodata.py
    data/geodata.py --> data/geodata/solar_2018.csv

    NREL_amy2018 --> data/weather.py
    data/weather.py --> data/weather/solar.csv
    data/weather/solar.csv --> data/geodata_project_years.py
    
    data/geodata/solar_2018.csv --> data/geodata_project_years.py
    data/geodata_project_years.py --> data/geodata/solar.csv
	USGS --> data/geodata/uspvdb.py
	USCB --> data/geodata/uspvdb.py
    data/geodata/solar.csv --> data/geodata/uspvdb.py
	data/geodata/uspvdb.py --> data/geodata/uspvdb.csv
```
## Wind Data

```mermaid
---
title: Wind Data Flow
---
graph TD

    NREL_amy2018 --> data/geodata.py
    data/geodata.py --> data/geodata/wind_2018.csv

    NREL_amy2018 --> data/weather.py
    data/weather.py --> data/weather/wind.csv
    data/weather/wind.csv --> data/geodata_project_years.py
    
    data/geodata/wind_2018.csv --> data/geodata_project_years.py
    data/geodata_project_years.py --> data/geodata/wind.csv
	USGS --> data/geodata/uswtdb.py
	USCB --> data/geodata/uswtdb.py
    data/geodata/wind.csv --> data/geodata/uswtdb.py
	data/geodata/uswtdb.py --> data/geodata/uswtdb.csv
```




# Validation

To review the load model, run the following marimo app:

	marimo run bob.py

Bob is the subject matter expert who can tell whether your solution is any good
just by looking at it.

# Notes

The `nodes.csv` contains a list of all the WECC 240 bus model locations with duplicate locations removed (see `nodes.py` for node reduction methodology).

# Data Sources
1. `powerplants.csv`: https://hifld-geoplatform.hub.arcgis.com/datasets/9dd630378fcf439999094a56c352670d_0/explore
2. WECC 240 bus model: https://www.nrel.gov/grid/assets/downloads/wecc-osl.zip. Citation: *Developing a Reduced 240-Bus WECC Dynamic Model for Frequency Response Study of High Renewable Integration, 2020 IEEE Power Engineering Society Transmission and Distribution Conference and Exposition (2020)*
3. `wecc_emissions.kml`: https://s3-us-west-1.amazonaws.com/widap.chassin.org/index.html
4. `caiso_co2_intensity_2021.csv`: https://www.electricitymaps.com/data-portal/united-states-of-america#data-portal-form
