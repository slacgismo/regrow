OPTIONS = --verbose
YEARS = 2018,2019,2020,2021,2022

all: powerplants weather

powerplants: powerplants.glm

powerplants.glm: powerplants.csv.zip
	python3 powerplants.py $< $@

weather: $(shell python3 -m weather --outputs)

$(shell python3 -m weather --outputs): $(shell python3 -m weather --inputs)
	python3 -m weather --update $(OPTIONS) --years=$(YEARS)
