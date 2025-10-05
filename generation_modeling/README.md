This folder contains the generation cost data used to scheduling.

# Setup

To setup an venv for this module do the following

    python3.12 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip -r requirements.txt

# Results

To obtain the pypower 'gen' and 'gencost' data arrays, do the following:

    import gendata
    print(gendata.model("generation_data.csv"))

Source: Jin Tan (NREL)

