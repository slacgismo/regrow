#
# WECC 240 model makefile
#
# Usage:
#	$ make [TARGET ...]
#

OPTIONS =
OPTIONS += --verbose

#
# Setup build environment
#
GLD_ETC = /usr/local/opt/gridlabd/current/share/gridlabd
FILES = 
INTERNAL =


# List all files that need to be updated when another file changes
# FILES += wecc240_psse.glm
# FILES += wecc240_gis.glm
# FILES += powerplants.glm
# FILES += powerlines.glm
# FILES += loads.glm
# FILES += geodata_commercial
FILES += weather_sensitivity

# List all files that need to be removed when cleaning up
INTERNAL += $(shell python3 -m geodata_commercial --internal)
# INTERNAL += $(shell python3 -m weather_sensitivity --internal)

#
# Targets
#
all: $(FILES)

clean:
	rm -i $(INTERNAL)

# wecc240_psse.glm: wecc240/240busWECC_2018_PSS.raw
# 	gridlabd convert $^ $@

# wecc240_gis.glm: wecc240_gis.py wecc240_gis.csv 
# 	gridlabd python $^ > $@

# powerplants.glm: powerplants.py powerplants.csv.zip
# 	gridlabd python $^ $@

# powerlines.glm: powerlines.py powerlines.csv.zip
# 	gridlabd python $^ $@

# loads.glm: res_loadshapes.csv.zip

# res_loadshapes.csv.zip: resstock.csv.zip res_loadshapes/floorareas.csv res_loadshapes.py
# 	gridlabd python res_loadshapes.py

# res_loadshapes/floorareas.csv: res_floorareas.py
# 	gridlabd python res_floorareas.py

# com_loadshapes/floorareas.csv: com_loadshapes.py
# 	gridlabd python com_loadshapes.py

# geodata_commercial: $(shell python3 -m geodata_commercial --outputs)

# $(shell python3 -m geodata_commercial --outputs): $(shell python3 -m geodata_commercial --inputs) geodata_commercial.py
# 	python3 -m geodata_commercial --update

weather_sensitivity: $(shell python3 -m weather_sensitivity --outputs)

$(shell python3 -m weather_sensitivity --outputs): $(shell python3 -m weather_sensitivity --inputs) weather_sensitivity.py
	python3 -m weather_sensitivity --update $(OPTIONS)
