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
    ["Alabama","AL","01"],
    ["Alaska","AK","02"],
    ["Arizona","AZ","04"],
    ["Arkansas","AR","05"],
    ["California","CA","06"],
    ["Colorado","CO","08"],
    ["Connecticut","CT","09"],
    ["Delaware","DE","10"],
    ["District of Columbia","DC","11"],
    ["Florida","FL","12"],
    ["Georgia","GA","13"],
    ["Hawaii","HI","15"],
    ["Idaho","ID","16"],
    ["Illinois","IL","17"],
    ["Indiana","IN","18"],
    ["Iowa","IA","19"],
    ["Kansas","KS","20"],
    ["Kentucky","KY","21"],
    ["Louisiana","LA","22"],
    ["Maine","ME","23"],
    ["Maryland","MD","24"],
    ["Massachusetts","MA","25"],
    ["Michigan","MI","26"],
    ["Minnesota","MN","27"],
    ["Mississippi","MS","28"],
    ["Missouri","MO","29"],
    ["Montana","MT","30"],
    ["Nebraska","NE","31"],
    ["Nevada","NV","32"],
    ["New Hampshire","NH","33"],
    ["New Jersey","NJ","34"],
    ["New Mexico","NM","35"],
    ["New York","NY","36"],
    ["North Carolina","NC","37"],
    ["North Dakota","ND","38"],
    ["Ohio","OH","39"],
    ["Oklahoma","OK","40"],
    ["Oregon","OR","41"],
    ["Pennsylvania","PA","42"],
    ["Puerto Rico","PR","72"],
    ["Rhode Island","RI","44"],
    ["South Carolina","SC","45"],
    ["South Dakota","SD","46"],
    ["Tennessee","TN","47"],
    ["Texas","TX","48"],
    ["Utah","UT","49"],
    ["Vermont","VT","50"],
    ["Virgin Islands","VI","78"],
    ["Virginia","VA","51"],
    ["Washington","WA","53"],
    ["West Virginia","WV","54"],
    ["Wisconsin","WI","55"],
    ["Wyoming","WY","56"],
]
state_codes_byname = dict([(x,{"usps":y,"fips":z}) for x,y,z in state_codes])
state_codes_byusps = dict([(y,{"name":x,"fips":z}) for x,y,z in state_codes])
state_codes_byfips = dict([(z,{"name":x,"usps":y}) for x,y,z in state_codes])

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

            print("name,usps,fips")
            print("\n".join([",".join(x) for x in state_codes]))

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
            for x,y,z in state_codes:
                m += testEq(name(y),x)
                m += testEq(name(z),x)
                m += testEq(usps(x),y)
                m += testEq(usps(z),y)
                m += testEq(fips(x),z)
                m += testEq(fips(y),z)
            if m.imag == 0:
                verbose(f"{int(m.real)} tests ok")
            else:
                error(f"{int(m.imag)} tests failed",E_TESTERR)

        elif arg in ["-U","--usps"]:

            print(",".join([usps(x) for x in sys.argv[n+1].split(",")]))
            n += 1

        elif arg in ["--verbose"]:

            VERBOSE = True

        else:
            raise Exception(f"option '{arg}' is not valid")

        n += 1

