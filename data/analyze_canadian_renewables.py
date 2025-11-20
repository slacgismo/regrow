import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import utils
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return (utils,)


@app.cell
def _(utils):
    canadian_renewables = utils.load_canadian_renewables_data()
    canadian_renewables
    return


if __name__ == "__main__":
    app.run()
