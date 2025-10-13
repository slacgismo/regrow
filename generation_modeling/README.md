This folder contains the generation cost data used to scheduling.

# Setup

To setup a `venv` environent for this module do the following

    python3.12 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip -r requirements.txt

# Results

To obtain the pypower `gen` and `gencost` data arrays, do the following:

    import gendata
    pp = gendata.model("generation_data.csv")

# Review

To review the results run the following marimo notebook

    marimo run generation_cost.py

Source: Jin Tan (NREL)

