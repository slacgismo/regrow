import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", app_title="REGROW Viewer")


@app.cell
def _():
    import sys
    import os
    import pandas as pd
    import boto3
    from datetime import datetime
    return boto3, datetime, os, pd, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # REGROW Forecast and Data Comparison Tool
    This tool will allow a user to select a particual WECC node from the table and it will then extract the targeted data products for plotting.
    """
    )
    return


@app.cell
def _(os, sys):
    script_dir = os.path.dirname(__file__) # Path of the current Marimo notebook
    print (script_dir)
    #parent_dir = os.path.join(script_dir, '../..') # Go up two levels to 'my_project'
    #target_dir = os.path.join(parent_dir, 'regrow_forecast_explorer/regrow_forecast_explorer/src') # Navigate to utility code
    config_dir = os.path.join(script_dir, 'config')

    # Add the target directory to sys.path
    #sys.path.insert(0, target_dir)
    sys.path.insert(0, config_dir)
    print(config_dir)

    #import REGROW files
    import forecast_explorer as fe
    import support as sp
    return fe, sp


@app.cell
def _(boto3):
    #Activate S3 access
    s3 = boto3.client('s3')
    s3_forecast_path = 'REGROW/Forecast_and_Actual_Weather_Merged/'
    return s3, s3_forecast_path


@app.cell(hide_code=True)
def _(mo):
    #collape
    mo.md(
        r"""
    ## Get WECC Node List
    Opens file stored in REGROW_forecast_explorer/config and extracts out the list of node IDs. Allow user to explore data table and then pick target node from list
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    <style>
    .output-wrapper {
        max-height: none !important;
        overflow: visible !important;
    }
    </style>
    """)

    print(f"Marimo version: {mo.__version__}")
    return (mo,)


@app.cell
def _(mo, sp):
    df_config = sp.read_config('C:/Projects/python/regrow_forecast_explorer/regrow_forecast_explorer/config/nodes.csv')
    node_table = mo.ui.table(df_config)
    node_table
    return (node_table,)


@app.cell(hide_code=True)
def _(node_table):
    # Get selected index labels
    selected_indices = node_table.value

    # Get the selected rows from the original DataFrame
    # Flatten if needed
    if isinstance(selected_indices, list) and any(isinstance(i, list) for i in selected_indices):
        # Flatten list of lists
        selected_indices = [item for sublist in selected_indices for item in sublist]

    # Wrap single selection in a list
    if not isinstance(selected_indices, list):
        selected_indices = [selected_indices]

    node_list = selected_indices[0]["geocode"].tolist()
    #print (node_list)
    return (node_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Extract Selected Node ID data file from S3""")
    return


@app.cell
def _(node_list, pd, s3, s3_forecast_path, sp):
    df = pd.DataFrame()
    if node_list:
        node_id=node_list[0]
        norm_file_base = '_forecast_actual_nsrdb_conus_2018-01-01-2022-12-31.csv'
        norm_file_path = s3_forecast_path + 'With_NSRDB_and_Conus/'
        target_filename = norm_file_path + node_list[0] + norm_file_base
        #print(norm_filename)
        df = sp.read_csv_from_s3(s3, "pvdrdb-transfer", target_filename) 
    return df, node_id


@app.cell(hide_code=True)
def _(df, mo):
    if df.empty:
        print ("Waiting...")
    else:
       # print('df active')
        # Show example data in table
        min_time = df['predict_time'].min()
        #print(min_time)
        filtered_df = df[df['predict_time'] == min_time]
        #print (filtered_df.shape)
        example_data_table = mo.ui.table(filtered_df)
        #print(example_data_table) 
    return


@app.cell(hide_code=True)
def _(mo):
    data_types = ['temperature', 'wind_speed', 'clouds']
    dt_dropdown = mo.ui.dropdown(data_types)
    resolution_types = [1, 4, 6, 12, 24]
    fp_dropdown = mo.ui.dropdown(resolution_types, value=1)
    analysis_list = ['norm']
    analysis_dropdown = mo.ui.dropdown(analysis_list)
    return analysis_dropdown, dt_dropdown, fp_dropdown


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### User selectable values for tuning analysis""")
    return


@app.cell(hide_code=True)
def _(datetime, df, mo):
    if df.empty:
        print ("Waiting...")

    checkbox_all_dates = mo.ui.checkbox(label="Full Date Range", value = False)

    picker_start_timestamp = mo.ui.datetime(label="Select date and time", value=datetime(2020, 8, 1, 00, 00, 00))
    picker_end_timestamp = mo.ui.datetime(label="Select date and time", value=datetime(2020, 8, 23, 23, 59, 59))
    mo.hstack([checkbox_all_dates, picker_start_timestamp, picker_end_timestamp])
    return checkbox_all_dates, picker_end_timestamp, picker_start_timestamp


@app.cell(hide_code=True)
def _(
    checkbox_all_dates,
    datetime,
    df,
    picker_end_timestamp,
    picker_start_timestamp,
):
    if df.empty:
        print ("Waiting...")

    if not checkbox_all_dates.value:
        start_timestamp = picker_start_timestamp.value
        end_timestamp = picker_end_timestamp.value
    else: 
        start_timestamp = df['forecast_time'].min()
        end_timestamp = df['forecast_time'].max()
        if isinstance(start_timestamp, datetime):
            start_timestamp = start_timestamp.tz_localize(None)
        else:
            dt = datetime.fromisoformat(start_timestamp)
            start_timestamp = dt.replace(tzinfo=None)
        if isinstance(end_timestamp, datetime):
            end_timestamp = end_timestamp.tz_localize(None)
        else:
            dt = datetime.fromisoformat(end_timestamp)
            end_timestamp = dt.replace(tzinfo=None)

    print (f"Using Date Range {start_timestamp} to {end_timestamp}")
    return end_timestamp, start_timestamp


@app.cell(hide_code=True)
def _(analysis_dropdown, df, dt_dropdown, fp_dropdown, mo):
    if df.empty:
        print ("Waiting...")

    selected_data_type = dt_dropdown.value
    selected_forecast_period = fp_dropdown.value
    selected_analysis = analysis_dropdown.value
    checkbox_forecast_scatter = mo.ui.checkbox(label="Prediction Forecast Data")
    checkbox_forecast_median = mo.ui.checkbox(label="Median Forecast Data")
    checkbox_noaa = mo.ui.checkbox(label="NOAA Actual Data")
    checkbox_nsrdb = mo.ui.checkbox(label="NSRDB Model Data")
    checkbox_conus = mo.ui.checkbox(label="Conus Model Data", value = True)

    # Wrap each dropdown with a label in a vertical stack
    dt_dropdown_with_label = mo.vstack(["Data Type", dt_dropdown])
    fp_dropdown_with_label = mo.vstack(["Forecast Resolution (h)", fp_dropdown])
    an_dropdown_with_label = mo.vstack(["Analysis Type", analysis_dropdown])
    checkboxes = mo.vstack([checkbox_forecast_scatter, checkbox_forecast_median, checkbox_noaa, checkbox_nsrdb, checkbox_conus])

    mo.hstack ([an_dropdown_with_label, dt_dropdown_with_label, fp_dropdown_with_label, checkboxes])
    return (
        checkbox_conus,
        checkbox_forecast_median,
        checkbox_forecast_scatter,
        checkbox_noaa,
        checkbox_nsrdb,
        selected_analysis,
        selected_data_type,
        selected_forecast_period,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Perform Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Raw Data at requested data resolution""")
    return


@app.cell
def _(
    checkbox_conus,
    checkbox_forecast_median,
    checkbox_forecast_scatter,
    checkbox_noaa,
    checkbox_nsrdb,
    df,
    end_timestamp,
    fe,
    node_id,
    pd,
    selected_data_type,
    selected_forecast_period,
    start_timestamp,
):
    # Plot direct comparison for values
    processed_df = pd.DataFrame()
    if df.empty:
        print ("Waiting...")
    else:
    #    if checkbox_conus.value:
    #        processed_df = fe.compare_linear_ground_vs_hub(node_id, 
    #                                        df, 
    #                                        selected_forecast_period,
    #                                        start_timestamp = start_timestamp,
    #                                        end_timestamp = end_timestamp,
    #                                        data_type=selected_data_type,
    #                                        show_scatter = checkbox_forecast_scatter.value
    #                                       )        
    #    else:
        processed_df = fe.compare_linear_forecast_model_actual(node_id, 
                                df, 
                                selected_forecast_period, 
                                start_timestamp = start_timestamp,
                                end_timestamp = end_timestamp,
                                data_type=selected_data_type,
                                show_forecast_scatter = checkbox_forecast_scatter.value,
                                show_forecast_median = checkbox_forecast_median.value,
                                show_noaa = checkbox_noaa.value,
                                show_nsrdb = checkbox_nsrdb.value,
                                show_conus = checkbox_conus.value
                               )

    return (processed_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Normalize""")
    return


@app.cell(hide_code=True)
def _(
    end_timestamp,
    fe,
    node_id,
    pd,
    processed_df,
    selected_analysis,
    selected_data_type,
    start_timestamp,
):
    norm_df = pd.DataFrame()
    if processed_df.empty:
        print('Waiting...')
    else:
        if not processed_df.empty:
            if selected_analysis == 'norm':
                norm_df = fe.normalize_linear_ground_vs_hub(
                            node_id, 
                            processed_df, 
                            norm_df,
                            start_timestamp = start_timestamp,
                            end_timestamp = end_timestamp,
                            data_type = selected_data_type, 
                            target_dir='', 
                            line_plot=True, 
                            regress_plot = False)
    return (norm_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Regression Analysis""")
    return


@app.cell(hide_code=True)
def _(
    df,
    end_timestamp,
    fe,
    node_id,
    norm_df,
    selected_analysis,
    selected_data_type,
    start_timestamp,
):
    if norm_df.empty:
        print ("Waiting...")
    else:
        # Do a regression plot if normalized available
        if not norm_df.empty:
            if selected_analysis == 'norm':
                fe.normalize_linear_ground_vs_hub(
                            node_id, 
                            df,
                            norm_df,
                            start_timestamp = start_timestamp,
                            end_timestamp = end_timestamp,
                            data_type = selected_data_type, 
                            target_dir='', 
                            line_plot=False, 
                            regress_plot = True)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
