"""GridLAB-D utilities

Syntax: (no CLI available)
"""
import os, sys
import json
import pandas as pd
import numpy as np
import math
import psm3 as pvlib_psm3
import datetime as dt
import psm3 as pvlib_psm3
import io, requests, zipfile, pdfplumber
import marimo as mo

WECC_GEN_TYPES = {
    
    "B": "Biomass",
    "NB": "Biomass",

    "G": "Gas",
    "DG": "Gas",
    "EG": "Gas",
    "TG": "Gas",
    "RG": "Gas",
    "SG": "Gas",
    "WG": "Gas",
    "NG": "Gas",
    "MG": "Gas",
    "CG": "Gas",
    
    "CE": "Geothermal",
    "NE": "Geothermal",
    
    "H": "Hydro",
    "NH": "Hydro",
    
    "N": "Nuclear",
    "NN": "Nuclear",
    
    "DP": "Solar",
    "S": "Solar",
    
    "C": "Steam",
    "E": "Steam",

    "W": "Wind",
    "NW": "Wind",
    "SW": "Wind",

    "R": "Renewable",

    "BA": "Battery",
    "BESS": "Battery",

    "PSH": "Pumphydro",

    "DC": "HVDC",
}

#
# Command args
#
def read_args(argv,docs=__doc__):
    """Process command line arguments for GridLAB-D run"""
    if len(argv) == 1:
        print("\n".join([x for x in docs.split("\n") if x.startswith("Syntax: ")]))
        exit(1)
    elif "-h" in argv or "--help" in argv or "help" in argv:
        print(__doc__)
        exit(0)
    elif "--verbose" in argv:
        options.verbose = True
        argv.remove("--verbose")
    elif "--debug" in argv:
        options.debug = True
        argv.remove("--debug")
    return argv[1:]


#
# General messaging to stderr
#
class options:
    context = '(no context)'
    verbose = False
    debug = True

E_OK = 0
E_NOENT = 2 # not found error
E_INTR = 4 # interrupted
E_INVAL = 22 # invalid argument

def error(code,msg):
    print(f"ERROR [{options.context}]: {msg}",file=sys.stderr)
    if type(code) is int:
        exit(code)
    elif type(code) is Exception:
        raise code

def warning(msg):
    print(f"WARNING [{options.context}]: {msg}",file=sys.stderr)

def verbose(msg,end="\n"):
    if options.verbose:
        print(msg,end=end,file=sys.stderr,flush=True)

#
# Geographic location encoding/decoding
#
_cache = {}

def _decode(geohash):
    """
    Decode the geohash to its exact values, including the error
    margins of the result.  Returns four float values: latitude,
    longitude, the plus/minus error for latitude (as a positive
    number) and the plus/minus error for longitude (as a positive
    number).
    """
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    __decodemap = { }
    for i in range(len(__base32)):
        __decodemap[__base32[i]] = i
    del i
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even: # adds longitude info
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:      # adds latitude info
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon, lat_err, lon_err

def geocode(geohash):
    """
    Decode geohash, returning two strings with latitude and longitude
    containing only relevant digits and with trailing zeroes removed.
    """
    if geohash in _cache:
        return _cache[geohash][0],_cache[geohash][1]
    lat, lon, lat_err, lon_err = _decode(geohash)
    from math import log10
    # Format to the number of decimals that are known
    lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
    lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)
    if '.' in lats: lats = lats.rstrip('0')
    if '.' in lons: lons = lons.rstrip('0')
    _cache[geohash] = (float(lats), float(lons))
    return float(lats), float(lons)

def geohash(latitude, longitude, precision=6):
    """Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    from math import log10
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    __decodemap = { }
    for i in range(len(__base32)):
        __decodemap[__base32[i]] = i
    del i
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)

def distance(a,b):
    """Get the distance between to geohashes"""
    lat1,lon1 = geocode(a)
    lat2,lon2 = geocode(b)
    return haversine_distance(lat1, lon1, lat2, lon2)

def distance2(a,b):
    """Get the distance squared between two geohashes"""
    x0,y0 = geocode(a)
    x1,y1 = geocode(b)
    dx,dy = x0-x1,y0-y1
    return dx*dx+dy*dy

def nearest(hash,hashlist,withdist=False):
    """Find the nearest geohash in a list of geohashes"""
    if len(hashlist) > 0:
        dist = sorted([(x,distance2(hash,x)) for x in hashlist],key=lambda y:y[1])
        return (dist[0][0],distance(hash,dist[0][0])) if withdist else dist[0][0]
    else:
        return (None,float('nan')) if withdist else None

def nearest2(test_latlon, latlonlist):
    test_lat, test_lon = test_latlon
    best_ix = 0
    best_dist = np.inf
    for _ix in range(len(latlonlist)):
        _lat, _lon = latlonlist[_ix]
        _new_dist = haversine_distance(_lat, _lon, test_lat, test_lon)
        if _new_dist < best_dist:
            best_dist = _new_dist
            best_ix = _ix
    return best_ix, latlonlist[best_ix], best_dist

#
# Calendar data
#
holidays = []
def is_workday(date,date_format="%Y-%m-%d %H:%M:%S"):
    global holidays
    if len(holidays) == 0:
        holidays = pd.read_csv("holidays.csv",
            index_col=[0],
            parse_dates=[0],
            date_format="%Y-%m-%d").sort_index()
    if type(date) is str:
        date = dt.datetime.strptime(date,date_format)
    if date.year < holidays.index.min().year or date.year > holidays.index.max().year:
        warning(f"is_workday(date='{date}',date_format='{date_format}') date is not in range of known holidays")
    return date.weekday()<5 and date not in holidays.index

#
# Weather data
#
def nsrdb_credentials(path="C:/Users/kperry/.nsrdb/credentials.json"): #os.path.join(os.environ["HOME"],".nsrdb","credentials.json")):
    try:
        with open(path,"r") as fh:
            return list(json.load(fh).items())[0]
    except Exception as err:
        error(E_INVAL,f"~/.nsrdb/credentials.json read failed - {err}")
        
        
def nsrdb_weather(location,year,
                  interval=30,
                  attributes={"solar[W/m^2]" : "ghi",
                              "temperature[degC]" : "air_temperature",
                              "wind[m/s]" : "wind_speed",
                              'dhi[W/m^2]': 'dhi',
                              'dni[W/m^2]': 'dni',
                              'winddirection[deg]': 'wind_direction',
                              'dewpoint[degC]': 'dew_point',
                              'relhumidity[pct]': 'relative_humidity',
                              'water[mm]': 'total_precipitable_water'
                              }):
    """
    Pull NSRDB data for a particular year and location. 
    
    Parameters
    ----------
    location: Str.
        Geohash of a particular location that will be decoded to get lat-long
        coordinates.
    year: Int.
        Year of data we want to pull data for.
    interval: Int.
        Frequency of data in minutes. Default 5
    attributes: Dictionary of string keys/values.
        Desired data fields to return as values, and final column names as keys.
        See pvlib documentaton for the full list of fields in NSRDB:
        https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.iotools.get_psm3.html
    
    Returns
    -------
    Pandas dataframe containing 'attribute' fields, with UTC ISO format
    datetime index.
    """
    lat,lon = geocode(location)
    leap = (year%4 == 0)
    email, api_key = nsrdb_credentials()
    # Pull from API and save locally
    psm3, _ = pvlib_psm3.get_psm3(lat, lon,
                                  api_key,
                                  email, year,
                                  attributes=attributes.values(),
                                  map_variables=True,
                                  interval=interval,
                                  leap_day=leap,
                                  timeout=60)
    cols_to_remove = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    psm3 = psm3.drop(columns=cols_to_remove)
    psm3.index = pd.to_datetime(psm3.index)
    psm3.rename(columns={"key_0": "datetime",
                         **{v: k for k, v in attributes.items()}},
                inplace=True)
    psm3 = psm3.round(3)  
    return psm3.sort_index()

NONASCII = {
    "\xe1" : "a",
    "\xe9" : "e",
    "\xed" : "i",
    "\xf1" : "n",
    "\xf3" : "o",
    "\xfc" : "u",
    # may need to add other someday
}

def strict_ascii(text):
    return ''.join([NONASCII[x] if x in NONASCII else x for x in text])

def haversine_distance(lat1, lon1, lat2, lon2):
    '''
    Returns the great-circle distance between two point in meters
    '''
    R = 6378.1e3 # radius of the earth in meters
    phi1 = lat1 * math.pi/180
    phi2 = lat2 * math.pi/180
    delta_phi = phi2 - phi1
    delta_lam = (lon2 - lon1) * math.pi/180
    a = (math.sin(delta_phi/2) * math.sin(delta_phi/2) 
         + math.cos(phi1) * math.cos(phi2) 
         * math.sin(delta_lam/2) * math.sin(delta_lam/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def load_reduced_network():
    network = pd.read_csv("wecc240_gis.csv", 
                      usecols=["Bus  Number","Bus  Name","Lat","Long"])
    network['geohash'] = network.apply(lambda row: geohash(row['Lat'], row['Long']), axis=1)
    grouped = network.groupby('geohash')
    reduced_network = grouped.first()
    reduced_network['node count'] = grouped.count()['Bus  Number'].values
    reduced_network['Bus  Number'] = grouped['Bus  Number'].apply(list)
    reduced_network['Bus  Name'] = grouped['Bus  Name'].apply(list)
    # classify renewable generation at nodes
    pv_node_geohashes = np.loadtxt('pv_node_geohashes.txt', dtype=str)
    wt_node_geohashes = np.loadtxt('wt_node_geohashes.txt', dtype=str)
    reduced_network['pv_gen'] = False
    reduced_network['wt_gen'] = False
    reduced_network.loc[pv_node_geohashes, 'pv_gen'] = True
    reduced_network.loc[wt_node_geohashes, 'wt_gen'] = True
    return reduced_network

def load_full_network():
    network = pd.read_csv("wecc240_gis.csv", 
                      usecols=["Bus  Number","Bus  Name","Lat","Long"])
    network['geohash'] = network.apply(lambda row: geohash(row['Lat'], row['Long']), axis=1)
    return network


def load_uspvdb():
    zipdata = zipfile.ZipFile(io.BytesIO(requests.get("https://energy.usgs.gov/uspvdb/assets/data/uspvdbCSV.zip").content))
    uspvdb = pd.read_csv(
        zipdata.open([x for x in zipdata.namelist() if x.endswith(".csv")][0],"r"),
        usecols = [
            "p_state", "p_county", "ylat", "xlong", "p_area", "p_name",
            "p_year", "p_tech_pri", "p_axis", "p_azimuth", "p_tilt",
            "p_battery", "p_cap_ac"
        ],
        
    )
    uspvdb.columns = [
        "state", "county", "latitude", "longitude", "area[m^2]", "name",
        "year", "gentype", "axis", "azimuth[deg]", "tilt[deg]", 
        "battery", "capacity[MW]"
    ]
    return uspvdb

def load_uswtdb():
    zipdata = zipfile.ZipFile(io.BytesIO(requests.get("https://energy.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip").content))
    uswtdb = pd.read_csv(
        zipdata.open([x for x in zipdata.namelist() if x.endswith(".csv")][0],"r"),
        usecols = [
            't_state', 't_county', 'p_name', 'p_year', 't_model', 't_cap', 
            't_hh', 't_rd', 'xlong', 'ylat'
        ],
        
    )
    uswtdb.columns = [
       "state", "county", "name", "year", "model", "capacity[MW]", "hub_height[m]", 
       "rotor_diameter[m]", "longitude", "latitude"
    ]
    uswtdb['county'] = uswtdb['county'].apply(lambda x: str(x)[:-7]) # remove " County" ending from each line
    uswtdb['capacity[MW]'] /= 1e3 # units in file are actually [kW], unlike uspvdb
    return uswtdb

def load_wecc_counties(fn='wecc_counties.csv'):
    if os.path.exists(fn):
        return pd.read_csv(fn, index_col=[0], header=0)
    print('Constructing county list from census.gov county info. This could take a couple minutes depending on webiste response...')
    census_url = "https://www2.census.gov/geo/docs/reference/state.txt"
    FIPS_STATES = pd.read_csv(
        census_url,
        delimiter="|",
        index_col=[1],
        usecols=[0,1,2],
        header=0,
        names=["fips","state","name"]
    ).to_dict('index')
    # Counties in WECC
    STATES = ["CA","WA","OR","ID","MT","WY","NV","UT","AZ","NM","CO"] 
    EXCLUDE = [ # Counties to leave out
        # MT
        '30019', # Daniels
        '30021', # Dawson
        '30025', # Fallon
        '30083', # Richland
        '30085', # Roosevelt
        '30091', # Sheridan
        '30109', # Wibaux
        # NM
        '35009', # Curry
        '35025', # Lea
        '35037', # Quay
        '35041', # Roosevelt
        ] 
    INCLUDE = {
        "TX": [
            '141', # El Paso
        ],
        "SD": [
            '033', # Custer
            '047', # Fall River
            '081', # Lawrence
        ]} # counties to add in from other states

    # Assemble county data
    counties_list = []
    for state in mo.status.progress_bar(STATES + list(INCLUDE)):
        fips = f"{FIPS_STATES[state]['fips']:02.0f}"

        # County population centroid data
        URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/"
        URL += f"CenPop2020_Mean_CO{fips}.txt"
        df = pd.read_csv(URL,
            converters = {
                "COUNAME": strict_ascii,
                "STATEFP": lambda x: f"{float(x):02.0f}",
                "COUNTYFP": lambda x: f"{float(x):03.0f}",
            },
            usecols = ["STATEFP","COUNTYFP","STNAME","COUNAME","POPULATION","LATITUDE","LONGITUDE"],
            )
        statename = df.STNAME.unique()[0].replace(' ','')
        df.columns = [x.lower() for x in df.columns]
        if state in INCLUDE:
            df.drop(
                df.loc[~df['countyfp'].isin(INCLUDE[state])].index,
                inplace=True,
                axis=0
            )
        rename = {"couname":"name"}
        df.columns = [rename[x] if x in rename else x for x in df.columns]
        df["fips"] = df["statefp"] + df["countyfp"]
        df["state"] = state
        df.drop(["statefp","countyfp"],axis=1,inplace=True)
        df.set_index("fips",inplace=True)
        # df["county"] = df.name + " " + df.state
        df.rename({"name":"county"},axis=1,inplace=True)
        df.drop(["stname","population"],axis=1,inplace=True)
        counties_list.append(df)
    wecc_counties = pd.concat(counties_list).drop(EXCLUDE,axis=0)
    wecc_counties.to_csv(fn)
    return wecc_counties

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    return dp[m][n]

def load_canadian_renewables_data(pdf_location="https://renewablesassociation.ca/wp-content/uploads/2025/01/New-Project-List.pdf"):
    tech_labels = ['Wind', 'Solar', 'Energy Storage', "Solar-Storage", "Wind-Storage"]
    provinces = ['NL', 'PE' ,'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
    pdf_response = requests.get(pdf_location)
    def my_float(_x):
        try:
            _o = float(_x)
        except ValueError:
            _o = np.nan
        return _o
    def my_int(_x):
        try:
            _o = int(_x)
        except ValueError:
            _o = np.nan
        return _o
    all_tables = []
    with io.BytesIO(pdf_response.content) as pdf_file:
        with pdfplumber.open(pdf_file) as pdf:
            for _page in pdf.pages:
                _tables = _page.extract_tables()
                all_tables.append(_tables)
    columns = all_tables[0][0][0]
    data = pd.DataFrame(columns=columns)
    _ix = 0
    for _p in range(len(all_tables)):
        for _r in range(len(all_tables[_p][0])):
            if _p == 0 and _r == 0:
                continue
            else:
                _v = all_tables[_p][0][_r]
                _label = _v[1]
                _distances = [levenshtein_distance(_label, _t) for _t in tech_labels]
                _new_label = tech_labels[np.argmin(_distances)]
                _province = _v[2]
                _distances = [levenshtein_distance(_province, _t) for _t in provinces]
                _new_p = provinces[np.argmin(_distances)]
                data.loc[_ix] = [_v[0], _new_label, _new_p, my_int(_v[3]), my_float(_v[4]), 
                                  my_float(_v[5]), my_float(_v[6]), 
                                  my_float(_v[7]), _v[8]]
                _ix += 1
    return data

def load_high_voltage_nodes():
    """
    These WECC nodes are transmission only and should not have generation or load attached.
    via David Chassin in discussion with Bennet Meyers 11/20/25
    """
    csv_str = """GEOHASH,BUS_I,NAME,BUS_TYPE,VOLTAGE,LOAD,GENERATION,GENOK
    9qhsdk,2603,VICTORVL,PQ,500.0,0.0,,0
    9qhsdk,2607,VICTORVL,PQ,287.0,0.0,,0
    9qq5wv,2901,ELDORADO,PQ,500.0,0.0,,0
    9q5zqv,2902,MOHAVE,PQ,500.0,0.0,,0
    9rg8bx,4003,BURNS,PQ,500.0,0.0,,0
    c21g7u,4007,CELILOCA,PQ,500.0,0.0,,0
    c21g7u,4010,CELILO,PQ,230.0,0.0,,0
    9r0vxp,8001,OLINDA,PQ,500.0,0.0,,0"""
    df = pd.read_csv(io.StringIO(csv_str), header=0)
    df['GEOHASH'] = df['GEOHASH'].apply(lambda x: x.strip())
    return df
