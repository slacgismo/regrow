"""
Pull down all of the EIA Form 860m data from the EIA website and process it.
"""

from bs4 import BeautifulSoup as bs
import requests, zipfile, io
import re
import pandas as pd
import os
import json
import datetime
import json

with open('energy_type_dict.json', 'r') as file:
    energy_type_dict = json.load(file)
    
with open('state_renamer_dict.json', 'r') as file:
    state_renamer_dict = json.load(file)
    

URL = "https://www.eia.gov/electricity/data/eia860m/"

def get_standardized_operating_year(df):
    df['planned_operating_year_month'] = (
        df['Planned Operation Year'].astype(str)
        + "-" + df['Planned Operation Month'].astype(str) + "-01")
    df['operating_year_month'] = (
        df['Operating Year'].astype(str) 
        + "-" + df['Operating Month'].astype(str) + "-01")
    df.loc[
        ((df['operating_year_month'] == "nan-nan-01") &
        (df['planned_operating_year_month'] != "nan-nan-01")),
        'operating_year_month'] = df['planned_operating_year_month']
    df['planned_retirement_year_month'] = (
        df['Planned Retirement Year'].astype(str) 
        + "-" + df['Planned Retirement Month'].astype(str) + "-01")
    df.loc[
        (df['planned_retirement_year_month'] == ' - -01') |
        (df['planned_retirement_year_month'] == 'nan-nan-01'),
        'planned_retirement_year_month'] = None
    df.loc[
        (df['operating_year_month'] == ' - -01') |
        (df['operating_year_month'] == 'nan-nan-01'),
        'operating_year_month'] = None
    return df['operating_year_month'], df['planned_retirement_year_month']

def get_soup(URL):
    return bs(requests.get(URL, verify=False).text, 'html.parser')

def pullXLSXFile(sheet_name, file_link):
    df = pd.read_excel("https://www.eia.gov"+ file_link, 
                        engine='openpyxl',
                        sheet_name=sheet_name)   
    index_cutoff = df[df[df.columns[0]] =='Entity ID'].index[0]
    df.columns = list(df.iloc[index_cutoff])
    df = df[df.index > index_cutoff]
    return df



if __name__ == "__main__":
    # Pull down the 860M data
    master_860m_data_pre = pd.DataFrame()
    for link in get_soup(URL).findAll("a", attrs={'href': re.compile(".xlsx")}):
        file_link = link.get('href')
        print(file_link)
        excel_file = pd.ExcelFile("https://www.eia.gov"+ file_link)
        sheet_names = excel_file.sheet_names
        try:
            if "Operating" in sheet_names:
                df_op = pullXLSXFile('Operating', file_link)
                df_op["file"] = file_link + "_op"
                df_op.columns = [x.replace("\n", "").lstrip() for x in df_op.columns]
                master_860m_data_pre = pd.concat([master_860m_data_pre, df_op])
            if "Canceled or Postponed" in sheet_names:
                df_cancel = pullXLSXFile('Canceled or Postponed', file_link)
                df_cancel["file"] = file_link + "_cancel"
                df_cancel.columns = [x.replace("\n", "").lstrip() for x in df_cancel.columns]
                master_860m_data_pre = pd.concat([master_860m_data_pre, df_cancel])
            if "Retired" in sheet_names:
                df_retired = pullXLSXFile('Retired', file_link)
                # Rename the retirement columns
                df_retired = df_retired.rename(columns={
                    'Retirement Month':'Planned Retirement Month',
                    'Retirement Year': 'Planned Retirement Year'})
                df_retired["file"] = file_link + "_retired"
                df_retired['Status'] = \
                    '(OS) Out of service and NOT expected to return to service in next calendar year'
                df_retired.columns = [x.replace("\n", "").lstrip() for x in df_retired.columns]
                master_860m_data_pre = pd.concat([master_860m_data_pre, df_retired])
        except Exception as e:
            print(e)
            print("ERROR ENCOUNTERED: " + file_link)
    master_860m_data = master_860m_data_pre.drop_duplicates()
    master_860m_data = master_860m_data[~master_860m_data['Plant Name'].isna()]
    # State renamer dictionary to standardize the state column
    master_860m_data['Plant State'] = master_860m_data[
        'Plant State'].map(state_renamer_dict).fillna(master_860m_data['Plant State'])
    # Get the latest reporting period for each entry (when the 860m data was generated)
    master_860m_data['report_year'] = [os.path.basename(
        x).split(".")[0].replace("_generator", " ").split(" ")[-1] for x in master_860m_data['file']]
    master_860m_data['report_year_max'] = master_860m_data.groupby("Plant ID")[
        'report_year'].transform("max")
    master_860m_data = master_860m_data[master_860m_data[
        'report_year'] == master_860m_data['report_year_max']]
    master_860m_data['report_date'] = [pd.to_datetime(os.path.basename(
        x).split(".")[0].replace("_generator", " 1, ").upper()) for x in master_860m_data['file']]
    master_860m_data['report_date_max'] = master_860m_data.groupby(["Plant ID", "Generator ID"])[
        'report_date'].transform("max")
    master_860m_data = master_860m_data[master_860m_data[
        'report_date'] == master_860m_data['report_date_max']]
    # Convert prime mover code to full name
    prime_mover_df = pd.read_csv("eia_energy_code_key.csv")
    master_860m_data = pd.merge(master_860m_data, prime_mover_df, 
                                on="Energy Source Code", how='left')
    master_860m_data = master_860m_data.rename(columns={"Grouping": "Prime Mover Group",
                                                        "Energy Source Description": "Prime Mover"})
    # Cleaned up data for insertion
    master_860m_data_clean = master_860m_data[['Entity ID', 'Entity Name', 'Plant ID', 
                                               'Plant Name', 'Plant State', 'County', 
                                               'Balancing Authority Code',
                                               'Sector', 'Unit Code', 'Technology',
                                               'Generator ID', 
                                               'Nameplate Capacity (MW)',
                                               'DC Net Capacity (MW)',
                                               'Net Summer Capacity (MW)',
                                               'Net Winter Capacity (MW)',
                                               'Operating Month',
                                               'Operating Year',
                                               'Planned Operation Month',
                                               'Planned Operation Year',
                                               'Planned Retirement Month',
                                               'Planned Retirement Year',   
                                               'Prime Mover', 
                                               "Prime Mover Group",
                                               'Status',
                                               'Latitude', 
                                               'Longitude',
                                               'report_date', 
                                               'file']].drop_duplicates()
    master_860m_data_clean = master_860m_data_clean.reset_index(drop=True)
    # Clean up the dataframe for DB insertion
    master_860m_data_clean['operating_year_month'], master_860m_data_clean[
        'planned_retirement_year_month'] = \
        get_standardized_operating_year(master_860m_data_clean)
    master_860m_data_clean = master_860m_data_clean.rename(columns={
        "Plant ID": "plant_id",
        "Generator ID": "generator_id",
        "Technology": "technology",
        'Nameplate Capacity (MW)': 'nameplate_capacity_mw',
        'report_date': "last_status_date",
        "Status": "status",
        'Plant Name': "plant_name",
        'Plant State': "state",
        'County': "county", 
        'Balancing Authority Code': "balancing_authority_code",
        'Latitude': "latitude", 
        'Longitude': "longitude",
        'Prime Mover': 'prime_mover',
        'Prime Mover Group': 'prime_mover_group',
        'Entity Name': 'utility_name',
        'Entity ID': 'entity_id',
        'Sector': 'sector',
        'Unit Code': 'unit_code',
        'DC Net Capacity (MW)': 'dc_net_capacity_mw', 
        'Net Summer Capacity (MW)': 'net_summer_capacity_mw',
        'Net Winter Capacity (MW)': 'net_winter_capacity_mw'
        })
    master_860m_data_clean['energy_source_desc'] = [
        energy_type_dict[x] if str(x) in energy_type_dict.keys() else None 
        for x in list(master_860m_data_clean['technology'])]
    master_860m_data_clean['status'] = [
        x.split("(")[-1].split(")")[0] if str(x) != "nan" else None 
        for x in list(master_860m_data_clean['status'])]
    master_860m_data_clean.loc[master_860m_data_clean['nameplate_capacity_mw'] == " ",
        'nameplate_capacity_mw'] = 0
    facilities = master_860m_data_clean[['plant_id', 'technology','energy_source_desc',
                                         'status', 'last_status_date', 'nameplate_capacity_mw',
                                         'operating_year_month', 'planned_retirement_year_month',
                                         'generator_id',
                                         'net_summer_capacity_mw','net_winter_capacity_mw',
                                         'dc_net_capacity_mw', 'prime_mover']].drop_duplicates()
    energy_types = master_860m_data_clean[['plant_id', 'technology','energy_source_desc',
                                           'status', 'last_status_date', 'nameplate_capacity_mw',
                                           'operating_year_month', 'planned_retirement_year_month',
                                           'generator_id','net_summer_capacity_mw','net_winter_capacity_mw',
                                           'dc_net_capacity_mw', 'prime_mover']].drop_duplicates()
    facilities.to_csv("facility_details.csv", index=False)
    energy_types.to_csv("energy_types.csv", index=False)