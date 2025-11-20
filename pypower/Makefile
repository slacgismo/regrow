FILES=
FILES+=hifld.py
FILES+=kml.py
FILES+=ppmodel.py
FILES+=psse.py
FILES+=psse2pp.py
FILES+=scheduling.py

DATA=wecc240_psse.raw powerplants.csv.zip wecc240_gis.csv
DATA+=$(wildcard wecc240/*.csv)
DATA+=$(wildcard wecc240/scheduling/*.csv)

SCRIPTS=summaries.csv

all: pylint.txt summaries

pylint.txt: $(FILES)
	pylint --fail-under=9 $(FILES) | tee pylint.txt

summaries: $(FILES) $(DATA) $(SCRIPT)
	python3 summaries.py
