import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __(pd):
    gis = pd.read_csv("wecc240_gis.csv")
    with open("wecc240_gis.glm","w") as _fh:
        for n,col in gis.iterrows():
            name = f"wecc240_N_{col.iloc[0]}"
            print(f"modify wecc240_psse_N_{col.iloc[0]}.latitude {col.Lat};",file=_fh)
            print(f"modify wecc240_psse_N_{col.iloc[0]}.longitude {col.Long};",file=_fh)
    return col, gis, n, name


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import json
    return json, mo, pd


if __name__ == "__main__":
    app.run()
