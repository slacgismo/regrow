# Load Model

The load model is generated for the nodes of the WECC model using the following steps:

1. Download 2018 load data from NREL data sources.
2. Map NSRDB weather data to county-level.
3. Project 2018 load data to other years using weather sensitivity.

This procedure gives us data that has the same errors relative to the temperature sensitivity model as the input year. This is a useful "quick and dirty" approximation of what the following years load might look like, and we will use this data to get the overall simulation up and running. We intend to revisit the question of how to estimate 2019–2022 (and especially 2020) load data in the future.

The load model and weather data are delivered in the `geodata` format required by [Arras Energy's gridlabd pypower geodata object](https://docs.gridlabd.us/_page.html?owner=arras-energy&project=gridlabd&branch=develop&folder=/Module/Pypower&doc=/Module/Pypower/Geodata.md). The load and weather data are stored in the `data/geodata` folder for use by the simulation.

## Load Geodata

The WECC load model is generated from [ResStock](https://resstock.nrel.gov/datasets) and [ComStock](https://comstock.nrel.gov/) databases using the 2018 actual meteorological year (AMY) datasets. The ResStock dataset includes county-level hourly total loads for the following residential building types:

  * Single-family attached
  * Single-family detached
  * Mobile homes
  * Multi-family 2-4 units
  * Multi-family 5+ units

The ComStock dataset includes county-level hourly total loads for the following commercial building types:

  * Full service restaurant
  * Hospital
  * Large hotel
  * Large office
  * Medium office
  * Outpatient health care
  * Primary school
  * Quick service restaurant
  * Retail standalone
  * Retail strip mall
  * Secondary school
  * Small hotel
  * Small office
  * Warehouse

The building load data is collected for each county in the WECC region using the `geodata.py` script. The county-level data is stored in temporarily in the `data/geodata/counties` folder but it is not preserved in the GitHub repository due to its large size and the relative ease of reloading it from NREL data sources if needed.

Use the following commands to regenerate the county-level building load geodata:

~~~
cd data
python3 -m geodata --update --refresh --verbose 
~~~

Geodata files contain data for a single property of the WECC model with location in columns named using [geohash codes](https://en.wikipedia.org/wiki/Geohash) and rows containing UTC timestamp, as required by GridLAB-D. The `geodata.py` script generates the following files in `data/geodata` for all building types combined:

  * `baseload.csv` - aggregate real power for end-uses that are not temperature sensitive
  * `heating.csv` - aggregate real power for end-uses that are sensitive to temperature during heating periods (when outdoor air temperature is below 10C)
  * `cooling.csv` - aggregate real power for end-uses that are sensitive to temperature during cooling periods (when outdoor air temperature is above 20C)
  * `total.csv` - aggregate real power for all end-uses combined.

The values for the heating and cooling cutoff temaperatures are stored in `data/config.py`.

This process can take up to several hours to run but if interrupted, the process will resume where it left off.

Note that the load model does not include industrial, agricultural, or public service loads. Consequently these must be added to the WECC simulation model separately.

## Weather Data

To project the 2018 building load data into other years, weather data for both the source and target years is required. The weather data from NSRDB is available in the `data/weather` folder. The `weather.py` script reorganizes the weather data so that it matches the structures of the county-level load geodata. The command to generate the weather geodata for the required year is

~~~
cd data
python3 -m weather --update --years=2018,2019,2020,2021,2022 --verbose
~~~

## Outyear Projections

The `geodata_project_years.py` script generates building load data for years other than the original 2018 NREL data source year.  The script computes for each hour of the target year the difference in real power corresponding to the difference in temperature observed between the source and target dataset.  The script also adjusts the source data to account for the shift in the day of week between the source and target yet.  

The temperature adjustment is based on the county-level linear weather sensitivity estimated by the `data/sensitivity.py` script and stored in the `data/sensitivity.csv`, e.g.,

  $P(t) = P(t-8736) + \max(0,T(t)-T(t-8736)) S(T(t)))$

where is $S(t)$ is the heating temperature sensitivity if $T(t)<T_{heat}$, the cooling temperature if $T(t)>T_{cool}$, otherwise zero.
 
To run the outyear projects, use the command:

~~~
cd data
python3 -m geodata_project_years
~~~

## Caveats and Future Work

1. The sensitivity analysis is only location-dependent and not time-dependent. A case can be made that weather sensitivity is time-dependent as well. Resolving this problem should be part of the load model project improvement should that be required.

2. The load projection model is very crude. It may not be sufficient for the needs of the study. If so, a more advanced but nonetheless efficient load model should be used, and may need to be developed.

