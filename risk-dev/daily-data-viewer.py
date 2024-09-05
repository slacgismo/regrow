import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    fb = mo.ui.file_browser(
        initial_path='PV-generator-data/',
        multiple=False
    )
    return fb,


@app.cell
def __(fb):
    fb
    return


@app.cell
def __(fb):
    fb.value[0].path
    return


@app.cell
def __(fb, pd):
    data = pd.read_csv(fb.value[0].path, index_col=0, parse_dates=[0])
    return data,


@app.cell
def __(data, mo):
    mo.ui.table(data)
    return


@app.cell
def __(data, mo, plt):
    data.plot(y=data.columns[0])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data, mo, plt):
    data.plot(y=data.columns[1])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data, mo, plt):
    data.plot(y=data.columns[2])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, pd, plt


if __name__ == "__main__":
    app.run()
