import marimo

__generated_with = "0.1.63"
app = marimo.App(width="full")


@app.cell
def __(pd):
    data = pd.read_csv("powerplants.csv",index_col=["STATE","ZIP","PRIM_FUEL"]).sort_index()
    return data,


@app.cell
def __(data, mo):
    mo.ui.table(data)
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    return mo, pd


if __name__ == "__main__":
    app.run()
