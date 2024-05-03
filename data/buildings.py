"""Generate load model"""

import os, sys
import pandas as pd

# PREFIX="wecc240_psse_"

TARGETS = {
	"P" : "../data/geodata/total.csv",
}

with open("buildings.glm","w") as glm:
	for target,source in TARGETS.items():
		print(f"""object geodata
{{
	file "{source}";
	target "load::{target}";
}}
""",file=glm)
