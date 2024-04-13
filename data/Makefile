#
# WECC 240 model makefile
#
# Usage:
#	$ make [TARGET ...]
#

#
# Setup build environment
#
GLD_ETC = /usr/local/opt/gridlabd/current/share/gridlabd
FILES = 

# List all files that need to be updated when another file changes
FILES += wecc240_psse.glm
FILES += wecc240_gis.glm
FILES += powerplants.glm

#
# Targets
#
all: $(FILES)

wecc240_psse.glm: wecc240/240busWECC_2018_PSS.raw
	gridlabd convert $^ $@

wecc240_gis.glm: wecc240_gis.py wecc240_gis.csv 
	gridlabd python $^ > $@

powerplants.glm: powerplants.py powerplants.csv.zip
	gridlabd python $^ $@

