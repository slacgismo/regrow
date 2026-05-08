import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return mo, pathlib, pd, plt


@app.cell(hide_code=True)
def _(mo):
    exp_dir_input = mo.ui.text(
        value="3years_6dayevent",
        label="experiment directory",
        full_width=True,
    )
    mo.vstack([mo.md("## Battery Sizing Results"), exp_dir_input])
    return (exp_dir_input,)


@app.cell
def _(exp_dir_input, pathlib, pd):
    _base = pathlib.Path(__file__).parent / exp_dir_input.value
    mpc_df = pd.read_csv(_base / "results.csv")
    one_shot_df = pd.read_csv(_base / "one_shot_results.csv")
    mpc_df
    return mpc_df, one_shot_df


@app.cell
def _(mpc_df, one_shot_df, plt):
    _metric_cols = [
        c for c in one_shot_df.columns if c not in ("Q",)
    ]
    _H_values = sorted(mpc_df["H"].unique())
    _n_cols = 3
    _n_rows = (len(_metric_cols) + _n_cols - 1) // _n_cols

    _fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(15, 4 * _n_rows))
    _axes = _axes.flatten()

    for _i, _metric in enumerate(_metric_cols):
        _ax = _axes[_i]
        for _H in _H_values:
            _sub = mpc_df[mpc_df["H"] == _H].sort_values("Q")
            _ax.plot(_sub["Q"], _sub[_metric], marker="o", label=f"MPC H={_H}")
        _os = one_shot_df.sort_values("Q")
        _ax.plot(_os["Q"], _os[_metric], marker="s", linestyle="--", color="black", label="one-shot")
        _ax.set_xlabel("Q [GWh]")
        _ax.set_title(_metric)
        _ax.set_xscale("log", base=2)
        _ax.legend(fontsize=7)

    for _j in range(_i + 1, len(_axes)):
        _axes[_j].set_visible(False)

    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
