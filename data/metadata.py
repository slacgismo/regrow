"""Process building model metadata

Syntax: python3 metadata.py [OPTIONS ...]

Options:
"""

import os, sys
import pandas as pd

VERBOSE = False
FRESHEN = False
METAFILE = "metadata.csv.zip"
METADIR = "metadata"
INDEXCOL = ["in.county"]
USECOLS = None

def split(metafile=METAFILE,
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

    if VERBOSE: print("Loading metadata",flush=True,end="...")
    meta = pd.read_csv(metafile,index_col=INDEXCOL)
    if VERBOSE: print("done",flush=True)

    for county in meta.index.get_level_values(0).unique():
        file = os.path.join(metadir,f"{county}.csv.zip")
        if not os.path.exists(file) or os.path.getctime(metafile) > os.path.getctime(file) or freshen:
            if VERBOSE: print(f"Saving {county}",flush=True,end="...")
            meta.loc[county].to_csv(file,index=True,header=True,compression="zip")
            if VERBOSE: print("done",flush=True)

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
            if VERBOSE: print(f"Reading {file}",flush=True,end="...")
            meta.append(pd.read_csv(os.path.join(metadir,file),**kwargs))
            if VERBOSE: print("done",flush=True)
    return pd.concat(meta)

def main(*args):
    n = 0
    while n < len(args):
        arg = args[n]
        if arg in ["-D","--directory"]:
            METADIR=args[n+1]
            n += 1
        elif arg in ["-F","--freshen"]:
            FRESHEN=True
        elif arg in ["-h","--help","help"]:
            print(__doc__)
        elif arg in ["-m","--merge"]:
            merge(args[n+1],index_col=INDEXCOL,usecols=USECOLS,low_memory=False).to_csv(index=True,header=True)
            n += 1
        elif arg in ["-s","--split"]:
            split(args[n+1])
        elif arg in ["-V","--verbose"]:
            VERBOSE=True
        else:
            raise Exception(f"invalid argument: {arg}")
        n += 1

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n".join([x for x in __doc__.split("\n") if x.startswith("Syntax: ")]))
    else:
        main(*sys.argv[1:])
