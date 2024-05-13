import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

INPUTS = {
	"GLOBALDATA" : "Global-Nuclear-Power-Tracker-October-2023.xlsx",
}

OUTPUTS = {
	"PLANTDATA" : "nuclear.csv",
}

STARTYEAR=2018
STOPYEAR=2022

data = pd.read_excel(INPUTS["GLOBALDATA"],sheet_name="Data",
	usecols=[1,2,3,6,7,8,9,10,11,12,23,24,25,26,27,]
	)

data.columns = ["country","name","unit_id","capacity[MW]","status",
	"reactor_type","reactor_model","start[y]","retirement[y]","planned_retirement[y]",
	"reference_capacity[MW]","design_capacity[MW]","thermal_capacity[MW]",
	"latitude","longitude",
]
data.drop(data[~data["status"].isin(["operating","retired"])].index,inplace=True)
data.drop(data[data["country"]!="United States"].index,inplace=True)
data.drop(data[data["retirement[y]"]<STARTYEAR].index,inplace=True)
data.drop(data[data["start[y]"]>STOPYEAR].index,inplace=True)
data.drop(["country","status"],axis=1,inplace=True)
data.to_csv(OUTPUTS["PLANTDATA"],header=True,index=False)