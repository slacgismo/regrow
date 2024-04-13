# powerplants.glm: powerplants.csv.zip
# 	python3 powerplants.py $< $@

GLD_ETC = /usr/local/opt/gridlabd/current/share/gridlabd

FILES = 
FILES += wecc240_psse.glm
FILES += wecc240_gis.glm

all: $(FILES)

wecc240_psse.glm: wecc240/240busWECC_2018_PSS.raw $(GLD_ETC)/raw2glm.py
	gridlabd convert $< $@

wecc240_gis.glm: wecc240_gis.py wecc240_gis.csv 
	gridlabd python $< > $@