# import eia 
import pandas as pd 
import requests 
import json


# def retrieve_time_series(api, series_ID):
#     """
#     Return the time series dataframe, based on API and unique Series ID
#     """
#     #Retrieve Data By Series ID 
#     series_search = api.data_by_series(series=series_ID)
#     ##Create a pandas dataframe from the retrieved time series
#     df = pd.DataFrame(series_search)
#     return df

def main():
    """
    Run main script
    """
    #Create EIA API using your specific API key
    api_key = ""
    api_url = "https://api.eia.gov/v2/electricity/retail-sales/data/?frequency=annual&data[0]=sales&facets[sectorid][]=COM&facets[sectorid][]=IND&facets[sectorid][]=OTH&facets[sectorid][]=RES&facets[sectorid][]=TRA&start=2019&end=2023&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(f"{api_url}&api_key={api_key}")
    # print(response.json())
    df = pd.DataFrame(response.json()["response"]["data"])
    print(df)
    with open('eia_all_sectors.json', 'w', encoding='utf-8') as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)
    df.to_csv('eia_all_sectors.csv', index=False)

if __name__== "__main__":
    main()
