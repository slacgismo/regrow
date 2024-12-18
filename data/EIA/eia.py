"""Read EIA energy use by state and sector"""
import os, sys
import pandas as pd 
import requests 
import json

pd.options.display.max_columns = None
pd.options.display.width = None

class NoApiKey(FileNotFoundError):
    pass

def main():
    """
    Run main script
    """
    try:
        apikey_file = os.path.join(os.getenv("HOME"),".eia","api_key")
        with open(apikey_file,"r") as fh:
            api_key = fh.read().strip()
    except FileNotFoundError as err:
        print(f"ERROR [eia.py]: You have not stored an API key from https://www.eia.gov/opendata/register.php in {apikey_file}",file=sys.stderr)
        raise
    api_url = "https://api.eia.gov/v2/electricity/retail-sales/data/?frequency=annual&data[0]=sales&facets[sectorid][]=COM&facets[sectorid][]=IND&facets[sectorid][]=OTH&facets[sectorid][]=RES&facets[sectorid][]=TRA&start=2018&end=2023&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    response = requests.get(f"{api_url}&api_key={api_key}")

    df = pd.DataFrame(response.json()["response"]["data"])
    df.drop(["stateDescription","sectorName","sales-units"],axis=1,inplace=True)
    df.columns = ["year","state","sector","energy"]
    df.fillna(0.0,inplace=True)
    df.set_index(["year","state","sector"],inplace=True)
    df.sort_index(inplace=True)

    with open('eia_all_sectors.json', 'w', encoding='utf-8') as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)
    df.to_csv('eia_all_sectors.csv', index=True)

if __name__== "__main__":
    main()
