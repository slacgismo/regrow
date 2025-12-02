FILES=
FILES+=eia860m.py
FILES+=hifld.py
FILES+=kml.py
FILES+=ppdata.py
FILES+=ppgen.py
FILES+=ppmodel.py
FILES+=psse.py
FILES+=psse2pp.py
FILES+=scheduling.py

DATA=wecc240_psse.raw
DATA+=wecc240_gis.csv
DATA+=powerplants.csv.zip
DATA+=$(wildcard wecc240/*.csv)
DATA+=$(wildcard wecc240/scheduling/*.csv)

SCRIPTS=summaries.csv

all: pylint.txt summaries

pylint.txt: $(FILES)
	pylint --fail-under=9 $(FILES) | tee pylint.txt

summaries: $(FILES) $(DATA) $(SCRIPT)
	python3 summaries.py

tests: $(FILES) $(DATA) tests.py
	python3 tests.py
