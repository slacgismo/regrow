To update the GLM file, run the Makefile.

# Load Model

## Buildings

The building load model is generated from the NREL ComStock [4] and ResStock [5] data for each county. The weather sensitivity for each county is then computed for both solar and temperature such that we have a least-square fit of $T$ observations

$L(t) = L_0(t) + 1_h(t) + a ~ S(t) + b ~ H(t) + c ~ C(t)$ for $h \in \{1,\cdots,48\}, t \in \{0,\cdots,T}$

where $h \in \{1,\cdots,24\}$ is hour of weekday and $ \in \{25,\cdots,48\}$ is hour of weekend, $L_h$ is the load at the $h$th hour, $1_h$ is the value 1 only if $t$ matches the hour $h$, $S$ is the solar irradiance, $H$ is the heating temperature difference, and $C$ is the cooling temperature difference. The coefficients $a$, $b$, and $c$ are the solar, heating, and cooling sensitivities. Then the base load is calculated as 

$L_h(t) = L(t) - a ~ S(t) - b ~ H(t) - c ~ C(t)$.

The load is then renormalized to match the annual energy reported in EIA [6] using the scaling factor

$E = \frac {E_{EIA}} {\Sigma_{t=1}^T L(t)}$

The building load models are stored as follows:

`geodata/commercial.csv`:
~~~
timestamp,geohash_1,geohash_2,...,geohash_N
YYYY-MM-DD HH:MM:SS+ZZZZ,L_1(0),L_2(0),...,L_N(0)
YYYY-MM-DD HH:MM:SS+ZZZZ,L_1(1),L_2(1),...,L_N(1)
...
YYYY-MM-DD HH:MM:SS+ZZZZ,L_1(T),L_2(T),...,L_N(T)
~~~

`geodata/commercial.glm`:
~~~
object geodata
{
  file "commercial.csv";
  target "load::P";
}
object load
{
  name "L_bus_1";
  parent "N_bus_1";
  latitude 37.5;
  longitude -122.5;
}
...
~~~

## Industrial

TODO

## Agricultural

TODO

## Transportation

TODO

# Data Sources
[1] `powerplants.csv`: https://hifld-geoplatform.hub.arcgis.com/datasets/9dd630378fcf439999094a56c352670d_0/explore

[2] WECC 240 bus model: https://www.nrel.gov/grid/assets/downloads/wecc-osl.zip. Citation: *Developing a Reduced 240-Bus WECC Dynamic Model for Frequency Response Study of High Renewable Integration, 2020 IEEE Power Engineering Society Transmission and Distribution Conference and Exposition (2020)*

[3] `wecc_emissions.kml`: https://s3-us-west-1.amazonaws.com/widap.chassin.org/index.html

[4] NREL ComStock Data, URL: https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fcomstock_amy2018_release_1%2Fweather%2Famy2018%2F

[5] NREL ResStock Data, URL: https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2024%2Fresstock_amy2018_release_2%2Fweather%2F
