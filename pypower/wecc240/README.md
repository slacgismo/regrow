# WECC 240 Model Data

This file is a manual extract of the `wecc240_psse.raw` and `WECC240_2018_Generation_scheduling.xlsx` files received from Jin Tan (jin.tan@nrel.gov).  

# Model Changes

The following modifications were made to this model to correct issues identified during the development of the WECC 240 model for REGROW.

1. Move WILSON bus 3405 to 37.290344,-120.404663 (9qdkhh).

2. Change PARKER bus type to 2 (PV instead of PQ) to connect HIFLD generation.

3. Converted CELILO/SYLMAR pseudo-loads to PDCI DC line

4. Converted ADELANTO/INTERMNT pseudo-loads to IMSST DC line

# Model Notes

There are three transformers that are simultaneous acting as lines:

1. PALOVRDE (1401@500kV) to PARKER (1403@230kV)

2. CORONADO (1101@500kV) to CHOLLA (1102@345kV) (double transformer)

3. CANALB (5002@500kV) to CA230TO (5003@230kV) 

The mapping tools will represented these transformers as lines to avoid adding extra busses to the model. We recognize that this will miss some transmission effects and constraints under certain conditions but we don't anticipate this to significant affect the results of the REGROW study.
