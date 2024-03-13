[![Simulation](https://github.com/slacgismo/regrow/actions/workflows/main.yml/badge.svg)](https://github.com/slacgismo/regrow/actions/workflows/main.yml)

# REGROW: Renewable Energy Generation Risk from Outlier Weather

This private repository is for team members of the REGROW project, funded by the U.S. Departement of Energy's Office of Electricity. Please use this to share important documents, notes, data, and code. 

The Technical Work Plan for the project is available in the top-level directory. Sub-directories are:

- `docs`: Documents, papers, etc. that are relevent to the research thrusts of the project. 

## Simulation Model

The GridLAB-D model is implemented in the [`model`](https://github.com/slacgismo/regrow/tree/main/model) folder. The main file is called [`wecc240.glm`](https://github.com/slacgismo/regrow/blob/main/model/wecc240.glm).  All supporting files included in this folder and all output files written to this folder will be included in the results download.  File in other folders, such as [`data`](https://github.com/slacgismo/regrow/tree/main/data), will not be included in the results download.

GitHub [`Actions`](https://github.com/slacgismo/regrow/actions) is used to start the simulation when the [`main`](https://github.com/slacgismo/regrow/tree/main) branch is updated or when a PR to [`main`](https://github.com/slacgismo/regrow/tree/main) is updated. To download the latest simulation results, see the [`Actions`](https://github.com/slacgismo/regrow/actions) tab above. For details on the [`Actions`](https://github.com/slacgismo/regrow/actions) workflow, see [`.github/workflows/main.yml`](https://github.com/slacgismo/regrow/blob/main/.github/workflows/main.yml).
