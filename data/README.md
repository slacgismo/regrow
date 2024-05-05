To update the GLM file, run the Makefile.

    make

# Generation model

Sources of `gencost` data

* [Flexible generation startup/shutdown costs](https://www.wecc.org/Reliability/1r10726%20WECC%20Update%20of%20Reliability%20and%20Cost%20Impacts%20of%20Flexible%20Generation%20on%20Fossil.pdf)

* [Solar PV plant modeling and validation](https://www.wecc.org/Reliability/Solar%20PV%20Plant%20Modeling%20and%20Validation%20Guidline.pdf)

* [NWPP production cost model](https://www.nwcouncil.org/2021powerplan_production-cost-simulation-results/)


# Load model

* [ResStock](https://www.nrel.gov/buildings/resstock.html)

* [ComStock](https://www.nrel.gov/buildings/comstock.html)

# Other data sources

1. `powerplants.csv`: https://hifld-geoplatform.hub.arcgis.com/datasets/9dd630378fcf439999094a56c352670d_0/explore

2. WECC 240 bus model: https://www.nrel.gov/grid/assets/downloads/wecc-osl.zip. Citation: *Developing a Reduced 240-Bus WECC Dynamic Model for Frequency Response Study of High Renewable Integration, 2020 IEEE Power Engineering Society Transmission and Distribution Conference and Exposition (2020)*

3. `wecc_emissions.kml`: https://s3-us-west-1.amazonaws.com/widap.chassin.org/index.html

# Validation

To review the load model, run the following marimo app:

	marimo run bob.py

Bob is the subject matter expert who can tell whether your solution is any good
just by looking at it.

