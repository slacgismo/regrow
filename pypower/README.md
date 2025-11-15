# PSS/E to PyPower Model Conversion

The data flow is as follows:

```flowchart LR
    wecc240_psse.raw --> manual_copy
    wecc240_psse.raw --> psse
    
    manual_copy --> wecc240_area.csv
    manual_copy --> wecc240_bus.csv
    manual_copy --> wecc240_branch.csv
    manual_copy --> wecc240_gen.csv
    manual_copy --> wecc240_load.csv
    manual_copy --> wecc240_shunt.csv
    manual_copy --> wecc240_xform.csv
    manual_copy --> wecc240_zone.csv

    manual_create --> wecc240_dcline.csv
    manual_create --> defaults.csv

    wecc240_area.csv --> psse
    wecc240_bus.csv --> pssex
    wecc240_branch.csv --> psse
    wecc240_xform.csv --> psse
    wecc240_load.csv --> psse
    wecc240_shunt.csv --> psse
    wecc240_gen.csv --> psse
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

- **Area** $\to$


