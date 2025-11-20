FILES=
FILES+=hifld.py
FILES+=kml.py
FILES+=ppmodel.py
FILES+=psse.py
FILES+=psse2pp.py
FILES+=scheduling.py

all: pylint summaries

pylint:
	pylint --fail-under=9 $(FILES)

summaries: summaries.py
	python3 summaries.py
