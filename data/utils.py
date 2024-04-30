import os, sys
import json
import pandas as pd
import math
import pvlib

#
# Command args
#
def read_args(argv,docs=__doc__):
    if len(argv) == 1:
        print("\n".join([x for x in docs.split("\n") if x.startswith("Syntax: ")]))
        exit(1)
    elif "-h" in argv or "--help" in argv or "help" in argv:
        print(__doc__)
        exit(0)
    elif "--verbose" in argv:
        options.verbose = True
        argv.remove("--verbose")
    return argv[1:]


#
# General messaging to stderr
#
class options:
    context = '(no context)'
    verbose = False

E_OK = 0
E_NOENT = 2 # not found error
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
    return math.sqrt(distance2(a,b))

def distance2(a,b):
    """Get the distance squared between two geohashes"""
    x0,y0 = geocode(a)
    x1,y1 = geocode(b)
    dx,dy = x0-x1,y0-y1
    return dx*dx+dy*dy

def nearest(hash,hashlist,withdist=False):
    """Find the nearest geohash in a list of geohashes"""
    if hashlist:
        dist = sorted([(x,distance2(hash,x)) for x in hashlist],key=lambda y:y[1])
        return (dist[0][0],distance(hash,dist[0][0])) if withdist else dist[0][0]
    else:
        return (None,float('nan')) if withdist else None

#
# Weather data
#
def nsrdb_credentials(path=os.path.join(os.environ["HOME"],".nsrdb","credentials.json")):
    try:
        with open(path,"r") as fh:
            return list(json.load(fh).items())[0]
    except Exception as err:
        error(E_INVAL,f"~/.nsrdb/credentials.json read failed - {err}")

def nsrdb_weather(location,year,interval=5,
                  attributes=['dhi',
                              'dni',
                              'ghi',
                              'wind_direction',
                              'wind_speed',
                              'air_temperature',
                              'dew_point',
                              'relative_humidity',
                              'total_precipitable_water']):
    """
    Pull NSRDB data for a particular year and location. 
    
    Parameters
    ----------
    location: Str.
    year: Int.
        Year of data we want to pull data for.
    interval: Int.
        Frequency of data in minutes. Default 5
    attributes: List of strings.
        Desired data fields to return. See pvlib documentaton for the
        full list:
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
    psm3, _ = pvlib.iotools.get_psm3(lat, lon,
                                    api_key,
                                    email, year,
                                    attributes=attributes,
                                    map_variables=True,
                                    interval=interval,
                                    leap_day=leap,
                                    timeout=60)
    cols_to_remove = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    psm3 = psm3.drop(columns=cols_to_remove)
    psm3.index = pd.to_datetime(psm3.index)
    psm3 = psm3.rename(columns={"key_0": "datetime"})
    psm3 = psm3.round(3)
    # Set datetime index to ISO UTC format
    psm3.index = psm3.index.tz_convert('UTC')
    psm3.index = psm3.index.map(lambda x: x.isoformat())
    return psm3.sort_index()
