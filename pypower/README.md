```mermaid
flowchart LR
    wecc240_psse.raw --> manual_copy

    manual_copy --> wecc240_area.csv
    manual_copy --> wecc240_bus.csv
    manual_copy --> wecc240_branch.csv
    manual_copy --> wecc240_gen.csv
    manual_copy --> wecc240_load.csv
    manual_copy --> wecc240_shunt.csv
    manual_copy --> wecc240_xform.csv
    manual_copy --> wecc240_zone.csv

    wecc240_bus.csv --> wecc240.py
    wecc240_branch.csv --> wecc240.py
    wecc240_gen.csv --> wecc240.py

    wecc240.py --> pypower
```