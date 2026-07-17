# Battery Data

This is your working directory. If you're a new contributor, welcome — this folder is yours to organize as the battery-related data and analysis work develops.

## What goes here

Battery system data and analysis code relevant to the REGROW project: raw datasets, processed/cleaned outputs, fixture files for testing, Python scripts, and Jupyter notebooks for data processing steps and numerical experiments. Examples:

- Battery capacity and specifications (by node, region, or technology class)
- Charge/discharge time series
- State-of-charge profiles
- Cost and degradation parameters
- Scripts that fetch, clean, or transform data
- Notebooks that run experiments or produce results figures

## Directory structure

There is no required internal structure yet. Add subdirectories as the work warrants. A `fixtures/` folder for small, stable test inputs is a common starting point:

```
data/batteries/
├── README.md          ← you are here
├── fixtures/          ← small, stable files used in tests or validation
├── raw/               ← data as downloaded, unmodified
├── processed/         ← cleaned or transformed outputs
└── notebooks/         ← Jupyter notebooks for experiments and figures
```

Feel free to add, rename, or reorganize — just keep this README up to date so the next person can orient quickly.

## Adding data files to git

Data files under the [GitHub 100 MB file size limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) can be committed directly and pushed to GitHub — no special tooling needed. For files above that limit, coordinate with the team (options include Git LFS or storing outside the repo and documenting the source here).

## Questions

Reach out to `bennetm [at] nlr [dot] gov` with any questions about the project or this dataset area.
