This folder contains the code and data needed to create the various PyPower models for the REGROW project.

# Quick Start

To create and run the `pypower` powerflow solution of a case, do the following:

    python3 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade pip -r requirements.txt
    python3
    from wecc240 import wecc240
    case = wecc240()
    from pypower.runpf import runpf
    runpf(case)

See [rwl/PYPOWER on GitHub](https://github.com/rwl/PYPOWER) for details on running PyPower cases.

# WECC240 Model Preparation

The WECC240 model in PyPower is prepared using the following high-level data flow:

```mermaid
flowchart LR

    subgraph sources
        wecc249_psse.raw --> psse.py
        WECC240_2018_Generation_schedule.xlsx --> scheduling.py
        powerplants.csv.zip --> hifld.py
        ResStock --> loads.py
        ComStock --> loads.py
        NREL --> renewables.py
    end

    subgraph modules
        direction TB
        psse.py --> psse2pp.py
        psse2pp.py --> ppmodel.py
        scheduling.py  --> ppmodel.py
        hifld.py --> ppmodel.py
        loads.py --> ppmodel.py
        renewables.py --> ppmodel.py
        ppmodel.py
    end

    ppmodel.py --> wecc240.py
```
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
    manual_create --> costs.csv

    wecc240_area.csv --> psse
    wecc240_branch.csv --> psse
    wecc240_bus.csv --> psse
    wecc240_gen.csv --> psse
    wecc240_gis.csv --> psse
    wecc240_load.csv --> psse
    wecc240_shunt.csv --> psse
    wecc240_xform.csv --> psse
    wecc240_zone.csv --> psse
    costs.csv --> psse2pp

    wecc240_dcline.csv --> psse
    subgraph wecc240.py
        psse --> psse2pp
        psse2pp --> ppmodel
    end

    ppmodel --> wecc240.case
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
    costs.csv --> pypower.gencost
    costs.csv --> pypower.dclinecost
    wecc240_dcline.csv --> pypower.dclinecost
    wecc240_dcline.csv --> pypower.dcline
```

#### DC Lines

In the original PSS/E model, the DC line are modeled as the following loads. 

| Bus Number | Bus Name | Load MW    | DC Line Name | Terminal |
| ---------: | -------- | ------:    | ------------ | -------- |
|       4010 | CELILO   |  2,904.493 | PDCI         | North    |
|       2619 | SYMLARLA | -2,466.528 | PDCI         | South    |
|       2600 | ADELANTO | -1,591.978 | IMSST        | South    |
|       2604 | INTERMT  |  1,791.945 | IMSST        | North    |

These loads have been replaced with DC lines in the PyPower model, as described in `wecc240_dcline.csv`. The constraints and losses for the DC lines are obtained from the corresponding references (see [Eriksson 2014](https://publisher.hitachienergy.com/download?DocumentID=9AKK106103A8918&LanguageCode=en&DocumentPartId=&Action=download&DocumentRevisionId=-&parentURL=68747470733a2f2f7075626c69736865722e68697461636869656e657267792e636f6d2f646f63756d656e74733f646f63547970653d416c6c25323046696c657326713d70616369666963253230696e746572746965) and [Wu 1988](https://ieeexplore.ieee.org/document/193910)).

| DC Line Name | Converter Loss | Line Loss | Voltage From | Voltage To | Minimum Power | Maximum Power | Minimum Reactive North | Maximum Reactive North | Minimum Reactive South | Maximum Reactive South |
| ------------ | -------------: | --------: | -----------: | ---------: | ------------: | ------------: | ---------------------: | ---------------------: | ----------------------: | ---------------------: |
| PDCI         |          20.77 |     1.38% |        1.075 |      1.012 |         -3100 |          3100 |                  -2000 |                   2000 |                  -2000 |                   2000 |
| IMSST        |             25 |     0.86% |        1.030 |      1.056 |             0 |          2400 |                   -100 |                    100 |                   -100 |                    100 |

The IMSST power ratings are based on the upgrade reported in the [Hitachi Project Summary](https://www.hitachienergy.com/us/en/news-and-events/customer-stories/intermountain-power-project). The voltage settings are imported from the PSS/E model. IMSST converter station losses were not found in the available public literature and are estimated based on PDCI converter station losses.

The DC line costs are listed in the `costs.csv`.

### PyPower Solvers

Three solver tests are performed on the resulting model:

- Powerflow (runpf)
- DC Optimal Powerflow (rundcopf)
- AC Optimal Powerflow (runopf)

## Modeling Options

External data can be included in the WECC240 model using the `options:list` keyword when calling the `wecc240()` method.  The following options are available:

#### `SCHEDULING`

Include the `SCHEDULING` option to update the case using the generation cost data from `WECC240_1018_Generation_schedule.xlsx` file extracted manually into the `wecc240_schedule_*.csv` files, which provide generator, line, and storage scheduling data.

The scheduling data is prepared and used to update cases as follows:

```mermaid
flowchart TD

    WECC240_1018_Generation_schedule.xlsx --> manual_copy
    manual_copy --> wecc240_scheduling_generator.csv
    manual_copy --> wecc240_scheduling_line.csv
    manual_copy --> wecc240_scheduling_storage.csv

    wecc240_scheduling_generator.csv --> Schedule
    wecc240_scheduling_line.csv --> Schedule
    wecc240_scheduling_storage.csv --> Schedule

    subgraph scheduling.py
        Schedule --> Schedule.update_case
        Schedule.update_case
    end

    Schedule.update_case --> wecc240.case
```

Note that this option is mutually exclusive with the `HIFLD` option.

#### `HILFD` (future work)

Include the `HIFLD` option to replace the generation fleet with the generators in the `powerplants.csv.zip` file.

#### `LOADS` (future work)

Include the `LOADS` option to replace the loads with the load model from NREL RESSTOCK and COMSTOCK loads. Note that using the load model requires the `datetime` option be specified.

#### `RENEWABLES` (future work)

Include the `RENEWABLES` optio to replace the renewable generation fleet with the NREL REGROW generation fleet.

## Result Check

The results of the powerflow solver as compared to the original input from PSS/E using the `voltage.png` and `voltage_errors.png`.  The former does a side-by-side comparison of each bus and the latter sorts the bus voltage and angle errors in descending order. This comparison is done for the AC powerflow of the original model compared to the original model (`tests/original_voltage_*.png`) and AC powerflow solution of the DC OPF solution compared to the original model (`tests/original_dcopf_voltage_*.png)`.

```mermaid
flowchart LR

    wecc240.case --> runpf
    wecc240.case --> rundcopf

    subgraph test.py
        
        runpf
        rundcopf --> runpf

    end
    runpf --> *_voltage.png
    runpf --> *_voltage_errors.png
```

# Code Checking

You can check the code using the `pylint` target of the `Makefile`, e.g.,

    make pylint
