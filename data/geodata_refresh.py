import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(lm):
    lm.County.counties(lm.County._regions["WECC"])
    return


@app.cell
def _():
    import marimo as mo
    import load_models as lm
    return (lm,)


if __name__ == "__main__":
    app.run()
