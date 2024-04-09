"""Process commercial building model metadata

Syntax: python3 com_metadata.py [OPTIONS ...]

Options:
    -D DIRECTORY    Set local folder in which files are stored
    -F              Force split to freshen local files
    -h|--help|help  Display this help
    -m|--merge FILE Merge local files into a single file
    -v|--verbose    Enable verbose output
"""

import os, sys
import pandas as pd
import config as cfg
import states

VERBOSE = False
FRESHEN = False
METAFILE = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/comstock_amy2018_release_1/metadata/baseline.parquet"
METADIR = "com_metadata"
INDEXCOL = ["in.nhgis_county_gisjoin"]
USECOLS = None
NOEXIT = False
DEBUG = False
QUIET = False
OUTPUT = "/dev/stdout"

def verbose(*args,**kwargs):
    if not "file" in kwargs:
        kwargs["file"] = sys.stderr
    if not "flush" in kwargs:
        kwargs["flush"] = True
    if VERBOSE:
        print(*args,**kwargs)

E_OK = 0
E_INVALID = 1
E_MISSING = 2
E_EXCEPTION = 9

def error(msg,exitcode=None,**kwargs):
    if not QUIET:
        if not "file" in kwargs:
            kwargs["file"] = sys.stderr
        if not "flush" in kwargs:
            kwargs["flush"] = True
        print(f"ERROR [metadata.py]: {msg}",**kwargs)
    if DEBUG:
        raise msg if type(msg) is Exception else Exception(msg)
    exit(exitcode)

def split(metafile,
          metadir=METADIR,
          freshen=FRESHEN,
          ) -> None:
    """Split the original metadata file into PUMA-level files

    Parameters:
    - metafile (str): the name of the original metadata file
    - metadir (str): the folder into which the metadata file is split
    - freshen (bool): force update of split files even if exists and up-to-date
    """
    os.makedirs(metadir,exist_ok=True)

    verbose(f"Reading metadata from {metafile}",end="...")
    if metafile.endswith(".parquet"):
        meta = pd.read_parquet(metafile).set_index(INDEXCOL)
    elif metafile.endswith(".csv") or metafile.endswith(".csv.zip"):
        meta = pd.read_csv(metafile,index_col=INDEXCOL)
    else:
        error("no metadata source given",exitcode=E_MISSING)

    n = 0
    county_list = [f"G{states.fips(x)}" for x in cfg.state_list]
    for county in meta.index.get_level_values(0).unique():
        if not county[0:3] in county_list:
            continue
        file = os.path.join(metadir,f"{county}.csv.zip")
        if not os.path.exists(file) or os.path.getctime(metafile) > os.path.getctime(file) or freshen:
            verbose(f"Saving {file}",end="...")
            meta.loc[county].to_csv(file,index=True,header=True,compression="zip")
            verbose("done")
            n += 1
    verbose(f"{n} files updated")

def merge(startswith="G",
          metadir=METADIR,
          **kwargs,
          ) -> pd.DataFrame:
    """Read and merge split metadata files

    Parameters:
    - startswith (str): The file name root to use when filtering which files to read
    - metadir (str): The folder in which the metadata files are stored
    - **kwargs: arguments to pass through to pd.read_csv
    """
    meta = []
    for file in os.listdir(metadir):
        if file.startswith(startswith):
            verbose(f"Reading {file}",end="...")
            meta.append(pd.read_csv(os.path.join(metadir,file),**kwargs))
            verbose("done")
    return pd.concat(meta)

def main(*args):
    n = 0
    global OUTPUT
    while n < len(args):
        arg = args[n]
        if arg in ["-D","--directory"]:
            global METADIR
            METADIR=args[n+1]
            n += 1
        elif arg in ["--debug"]:
            global DEBUG
            DEBUG=True
        elif arg in ["-F","--freshen"]:
            global FRESHEN
            FRESHEN=True
        elif arg in ["-h","--help","help"]:
            print(__doc__)
        elif arg in ["-m","--merge"]:
            with open(OUTPUT,"wt") as fh:
                merge(args[n+1],index_col=INDEXCOL,usecols=USECOLS,low_memory=False).to_csv(OUTPUT,index=True,header=True)
                n += 1
        elif arg in ["-o","--output"]:
            OUTPUT=arg[n+1]
            n += 1
        elif arg in ["-s","--split"]:
            split(args[n+1] if n < len(args)-1 else METAFILE)
            n += 1
        elif arg in ["--silent"]:
            global QUIET
            QUIET=True
        elif arg in ["--verbose"]:
            global VERBOSE
            VERBOSE=True
        else:
            raise Exception(f"invalid argument: {arg}")
        n += 1

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n".join([x for x in __doc__.split("\n") if x.startswith("Syntax: ")]))
    else:
        try:
            main(*sys.argv[1:])
        except SystemExit:
            pass
        except Exception as err:
            if DEBUG:
                raise
            error(err)
