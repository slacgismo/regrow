"""State USPS and FIPS codes

Syntax: python3 states.py [OPTIONS]

Options:

    --debug               Enable exceptions instead of errors
    --dump                Output codes data in CSV format
    -F|--fips STR[,...]   Lookup FIPS code for state name of USPS codes
    -h|--help|help        Output this help
    -N|--name STR[,...]   Lookup state name for FIPS or USPS codes
    --test                Run self-test
    -U|--usps STR[,...]   Lookup USPS code for state name or FIPS codes
    --verbose             Enable verbose output

"""

import os, sys
import pandas as pd

VERBOSE = False
QUIET = False
DEBUG = False

def verbose(msg,**kwargs):
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if VERBOSE:
        print(f"VERBOSE [{os.path.basename(sys.argv[0])}]: {msg}",**kwargs)

E_OK = 0
E_INVALID = 1
E_MISSING = 2
E_TESTERR = 8
E_EXCEPTION = 9

def error(msg,exitcode,**kwargs):
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if DEBUG:
        raise exitcode if type(exitcode) is Exception else Exception(f"ERROR [{os.path.basename(sys.argv[0])}]: {exitcode} - {msg}")
    elif not QUIET:
        print(f"ERROR [{os.path.basename(sys.argv[0])}]: {msg} (exitcode)",**kwargs)
    exit(int(exitcode))

state_codes = [
    ["Alabama","AL","01",-6],
    ["Alaska","AK","02",-9],
    ["Arizona","AZ","04",-7],
    ["Arkansas","AR","05",-6],
    ["California","CA","06",-8],
    ["Colorado","CO","08",-7],
    ["Connecticut","CT","09",-5],
    ["Delaware","DE","10",-5],
    ["District of Columbia","DC","11",-5],
    ["Florida","FL","12",-5],
    ["Georgia","GA","13",-5],
    ["Hawaii","HI","15",-9],
    ["Idaho","ID","16",-7],
    ["Illinois","IL","17",-6],
    ["Indiana","IN","18",-5],
    ["Iowa","IA","19",-6],
    ["Kansas","KS","20",-6],
    ["Kentucky","KY","21",-6],
    ["Louisiana","LA","22",-6],
    ["Maine","ME","23",-5],
    ["Maryland","MD","24",-5],
    ["Massachusetts","MA","25",-5],
    ["Michigan","MI","26",-5],
    ["Minnesota","MN","27",-6],
    ["Mississippi","MS","28",-6],
    ["Missouri","MO","29",-6],
    ["Montana","MT","30",-7],
    ["Nebraska","NE","31",-6],
    ["Nevada","NV","32",-7],
    ["New Hampshire","NH","33",-5],
    ["New Jersey","NJ","34",-5],
    ["New Mexico","NM","35",-7],
    ["New York","NY","36",-5],
    ["North Carolina","NC","37",-5],
    ["North Dakota","ND","38",-6],
    ["Ohio","OH","39",-5],
    ["Oklahoma","OK","40",-6],
    ["Oregon","OR","41",-8],
    ["Pennsylvania","PA","42",-5],
    ["Puerto Rico","PR","72",-4],
    ["Rhode Island","RI","44",-5],
    ["South Carolina","SC","45",-5],
    ["South Dakota","SD","46",-6],
    ["Tennessee","TN","47",-6],
    ["Texas","TX","48",-6],
    ["Utah","UT","49",-7],
    ["Vermont","VT","50",-5],
    ["Virgin Islands","VI","78",-4],
    ["Virginia","VA","51",-5],
    ["Washington","WA","53",-8],
    ["West Virginia","WV","54",-5],
    ["Wisconsin","WI","55",-6],
    ["Wyoming","WY","56",-7],
]
state_codes_byname = dict([(x,{"usps":y,"fips":z,"tz":w}) for x,y,z,w in state_codes])
state_codes_byusps = dict([(y,{"name":x,"fips":z,"tz":w}) for x,y,z,w in state_codes])
state_codes_byfips = dict([(z,{"name":x,"usps":y,"tz":w}) for x,y,z,w in state_codes])

def name(state,exception=False):
    """Lookup state name from USPS or FIPS code"""
    try:
        if state in state_codes_byusps:
            return state_codes_byusps[state]["name"]
        elif state in state_codes_byfips:
            return state_codes_byfips[state]["name"]
    except:
        if exception:
            raise
    return None

def fips(state,exception=False):
    """Lookup FIPS from USPS code or name"""
    try:
        if state in state_codes_byusps:
            return state_codes_byusps[state]['fips']
        elif state in state_codes_byname:
            return state_codes_byname[state]['fips']
    except:
        if exception:
            raise
    return None

def usps(state,exception=False):
    """Lookup USPS from FIPS code or name"""
    try:
        if state in state_codes_byname:
            return state_codes_byname[state]["usps"]
        elif state in state_codes_byfips:
            return state_codes_byfips[state]["usps"]
    except:
        if exception:
            raise
    return None

def timezone(state,exception=False):
    """Lookup timezone from FIPS code or name"""
    try:
        if state in state_codes_byusps:
            return state_codes_byusps[state]['tz']
        elif state in state_codes_byname:
            return state_codes_byname[state]["tz"]
        elif state in state_codes_byfips:
            return state_codes_byfips[state]["tz"]
    except:
        if exception:
            raise
    return None

def testEq(a,b):
    """Test equality of results"""
    if not a == b:
        verbose(f"{a} == {b} failed")
        return 1j # failure
    else:
        return 1 # success

if __name__ == "__main__":

    if len(sys.argv) == 1:

        print("\n".join([x for x in __doc__.split("\n") if x.startswith("Syntax: ")]))
        exit(0)

    n = 1
    while n < len(sys.argv):

        arg = sys.argv[n]
        if arg in ["--debug"]:

            DEBUG = True

        elif arg in ["--dump"]:

            print("name,usps,fips,tz")
            print("\n".join([",".join(map(str,x)) for x in state_codes]))

        elif arg in ["-F","--fips"]:

            print(",".join([fips(x) for x in sys.argv[n+1].split(",")]))
            n += 1

        elif arg in ["-h","--help","help"]:
            
            print(__doc__)

        elif arg in ["-N","--name"]:

            print(",".join([name(x) for x in sys.argv[n+1].split(",")]))
            n += 1

        elif arg in ["--silent"]:

            QUIET = True

        elif arg in ["--test"]:

            m = 0j
            for x,y,z,w in state_codes:
                m += testEq(name(y),x)
                m += testEq(name(z),x)
                m += testEq(usps(x),y)
                m += testEq(usps(z),y)
                m += testEq(fips(x),z)
                m += testEq(fips(y),z)
                m += testEq(timezone(x),w)
                m += testEq(timezone(y),w)
                m += testEq(timezone(z),w)
            if m.imag == 0:
                verbose(f"{int(m.real)} tests ok")
            else:
                error(f"{int(m.imag)} tests failed",E_TESTERR)

        elif arg in ["-U","--usps"]:

            print(",".join([usps(x) for x in sys.argv[n+1].split(",")]))
            n += 1

        elif arg in ["-Z","--timezone"]:

            print(",".join([str(timezone(x)) for x in sys.argv[n+1].split(",")]))
            n += 1

        elif arg in ["--verbose"]:

            VERBOSE = True

        else:
            raise Exception(f"option '{arg}' is not valid")

        n += 1

