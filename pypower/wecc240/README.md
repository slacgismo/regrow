# WECC 240 Model Data

This file is a manual extract of the `wecc240_psse.raw` and `WECC240_2018_Generation_scheduling.xlsx` files received from Jin Tan (jin.tan@nrel.gov).  

# Model Changes

The following modifications were made to this model to correct issues identified during the development of the WECC 240 model for REGROW.

1. Move WILSON bus 3405 to 37.290344,-120.404663 (9qdkhh).

2. Change PARKER bus type to 2 (PV instead of PQ) to connect HIFLD generation.

3. Converted CELILO/SYLMAR pseudo-loads to PDCI DC line

4. Converted ADELANTO/INTERMNT pseudo-loads to IMSST DC line
