# PSS/E to PyPower Model Conversion

The data flow is as follows:

```mermaid
flowchart LR
    wecc240_psse.raw --> manual_copy
    wecc240_psse.raw --> psse
    
    manual_copy --> wecc240_area.csv
    manual_copy --> wecc240_branch.csv
    manual_copy --> wecc240_bus.csv
    manual_copy --> wecc240_gen.csv
    manual_copy --> wecc240_load.csv
    manual_copy --> wecc240_shunt.csv
    manual_copy --> wecc240_xform.csv
    manual_copy --> wecc240_zone.csv

    manual_create --> wecc240_dcline.csv
    manual_create --> defaults.csv

    wecc240_area.csv --> psse
    wecc240_branch.csv --> psse
    wecc240_bus.csv --> pssex
    wecc240_gen.csv --> psse
    wecc240_load.csv --> psse
    wecc240_shunt.csv --> psse
    wecc240_xform.csv --> psse
    wecc240_zone.csv --> psse
    defaults.csv --> psse2pp

    wecc240_dcline.csv --> psse
    subgraph wecc240.py
        psse --> psse2pp
        psse2pp --> ppmodel
    end

    ppmodel --> case
    case --> runpf
    case --> rundcopf
    case --> runopf

    subgraph pypower
        runpf
        rundcopf
        runopf
    end

    runpf --> results
    rundcopf --> results
    runopf --> results
```

## Methodology

### PSSE Segments

The PSSE data segments are extract manually as follows:

- **AREA** $\to$ `wecc240_area.csv`
- **BRANCH** $\to$ `wecc240_branch.csv` $\to$ `psse.branch`
- **BUS** $\to$ `wecc240_bus.csv` $\to$ `psse.bus`
- **GEN** $\to$ `wecc240_gen.csv` $\to$ `psse.gen`
- **LOAD** $\to$ `wecc240_load.csv` $\to$ `psse.load`
- **SHUNT** $\to$ `wecc240_shunt.csv` $\to$ `psse.shunt`
- **XFORM** $\to$ `wecc240_xform.csv` $\to$ `psse.xform`
- **ZONE** $\to$ `wecc240_zone.csv`

The segment files must be edited manually to clean the header names and remove extra whitespaces in strings. In addition, the transformer segment (`XFORM`) must be edited to merge the multiline entries and remove the extra columns that are not included in the data segment.

A segment for `DCLINE` must be manual constructed to provide the real definitions of the PDCI and Intermountain SST lines. The corresponding negative and positive loads at busses 4007 and 2619 (PDCI) and 2600 and 2601 (Intermountain SST) must also be removed manually from the load segment.

### PSSE to PP converter

The `psse2pp` converter converts the PSSE segments into pypower blocks as follows:

```mermaid
flowchart LR

    psse.bus --> pypower.bus
    psse.load --> pypower.bus
    psse.shunt --> pypower.bus
    
    psse.branch --> pypower.branch
    psse.xform --> pypower.branch

    psse.gen --> pypower.gen
    psse.gen --> pypower.gencost

    subgraph psse
        psse.branch
        psse.xform
        psse.bus
        psse.load
        psse.shunt
        psse.gen
    end

    subgraph pypower
        pypower.branch
        pypower.bus
        pypower.gen
        pypower.gencost
        pypower.dcline
        pypower.dclinecost
    end
    defaults.csv --> pypower.gencost
    wecc240_dcline.csv --> pypower.dcline
    wecc240_dcline.csv --> pypower.dclinecost
```

### PyPower Solvers

Three solver tests are performed on the resulting model:

- Powerflow (runpf)
- DC Optimal Powerflow (rundcopf)
- AC Optimal Powerflow (runopf)
