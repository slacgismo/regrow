"""
Runs all the data processing files to get raw data into one scada file or
multiple turbine level files with standardized names and dtypes.


All data processing follows these step:
    1) Combine owner provided data into one scada file if multiple files are
    provided.
    2) Standardize the column names with the following:
        datetime: (datetime) tz aware datetime
        turbine: (str) turbine's name or id
        wind_speed_ms: (float) turbine's wind speed in meters per second
        power_kW: (float) tubrine's power output in kW
        yaw_deg: (float) nacelle position or yaw in degrees, if available
        wind_dir_deg: (float) wind direction in degrees, if available
        
        Note: Only o46 has both yaw_deg and wind_dir_deg. n23 have only
        wind_dir_deg and all other sites only have yaw_deg.
            
    3) Process data if box folder contains any data processing scripts.
    4) Generate turbine level files for easier analysis and plotting.

"""

from data_processing_scripts.o46_data_processing import o46DataProcessing
from data_processing_scripts.k64a_k64b_data_processing import k64DataProcessing
from data_processing_scripts.c43_data_processing import c43DataProcessing
from data_processing_scripts.u15_x41_data_processing import u15x41DataProcessing
from data_processing_scripts.n23_data_processing import n23DataProcessing
from data_processing_scripts.r47_data_processing import r47DataProcessing
from data_processing_scripts.t75_data_processing import t75DataProcessing
from data_processing_scripts.s33_data_processing import s33DataProcessing
from data_processing_scripts.q56_data_processing import q56DataProcessing
from data_processing_scripts.w34_data_processing import w34DataProcessing

scada_save_folder = "./data_processing_scripts/SCADA/"
turbine_save_folder = "./data_processing_scripts/SCADA_reformatted/"

run=True
if run:
    # o46: Avangrid baffin
    o46 = o46DataProcessing("./data_processing_scripts/raw_data_dump/"+
                            "o46/")
    o46.scada_processing(scada_save_folder)
    o46.generate_turbine_lvl_files(turbine_save_folder)
    print("o46 data processing finished")
    #---------------------------------------------------------------------
    
    # c43: EDFRE spinning spur 1
    c43 = c43DataProcessing("./data_processing_scripts/raw_data_dump/c43/" +
                            "Spur1_SCADA_10min_Since_7_13_15.csv")
    c43.scada_processing(scada_save_folder)
    c43.generate_turbine_lvl_files(turbine_save_folder)
    print("c43 data processing finished")
    
    # k64a: EDFRE kelly creek
    k64 = k64DataProcessing("./data_processing_scripts/raw_data_dump/k64/" +
                            "SCADA_Kelly Creek_20170101-20200501.csv")
    k64.scada_processing(scada_save_folder)
    k64.generate_turbine_lvl_files(turbine_save_folder)
    print("k64a data processing finished")
    
    # k64b: EDFRE Pilot Hill
    k64 = k64DataProcessing("./data_processing_scripts/raw_data_dump/k64/" +
                            "SCADA_Pilot Hill_20151001-20200501.csv")
    k64.scada_processing(scada_save_folder)
    k64.generate_turbine_lvl_files(turbine_save_folder)
    print("k64b data processing finished")
    #---------------------------------------------------------------------

    
    # u15: Pattern St. joseph    
    pattern = u15x41DataProcessing("./data_processing_scripts/raw_data_dump/" + 
                                   "u15/SJW_SCADA.csv")
    pattern.scada_processing(scada_save_folder)
    pattern.generate_turbine_lvl_files(turbine_save_folder)
    print("u15 data processing finished")

    # x41: Pattern panhandle 2
    pattern = u15x41DataProcessing("./data_processing_scripts/raw_data_dump/" + 
                                   "x41/Pan2_SCADA.csv")
    pattern.scada_processing(scada_save_folder)
    pattern.generate_turbine_lvl_files(turbine_save_folder)
    print("x41 data processing finished")
    #---------------------------------------------------------------------
    

    # n23: ENEL origin
    n23 = n23DataProcessing("./data_processing_scripts/raw_data_dump/n23/")
    n23.scada_processing(scada_save_folder)
    n23.generate_turbine_lvl_files(turbine_save_folder)
    print("n23 data processing finished")
    #---------------------------------------------------------------------
    
    
    # r47: EDPR prairie star
    r47 = r47DataProcessing("./data_processing_scripts/raw_data_dump/r47/")
    r47.scada_processing(scada_save_folder)
    r47.generate_turbine_lvl_files(turbine_save_folder)
    print("r47 data processing finished")
    
    # t75: EDPR meridian way
    t75 = t75DataProcessing("./data_processing_scripts/raw_data_dump/t75/" +
                            "MeridanWay_I-Turbine_data.csv")
    t75.scada_processing(scada_save_folder)
    t75.generate_turbine_lvl_files(turbine_save_folder)
    print("t75 data processing finished")
    
    # s33: EDPR los mirasoles
    s33 = s33DataProcessing("./data_processing_scripts/raw_data_dump/s33/" +
                            "Los Mirasoles-Scada_data.csv")
    s33.scada_processing(scada_save_folder)
    s33.generate_turbine_lvl_files(turbine_save_folder)
    print("s33 data processing finished")
    #---------------------------------------------------------------------
    
    
    # q56: E-ON champion
    q56 = q56DataProcessing("./data_processing_scripts/raw_data_dump/q56/")
    q56.scada_processing(scada_save_folder)
    q56.generate_turbine_lvl_files(turbine_save_folder)
    print("q56 data processing finished")
    
    # w34: E-ON pioneer
    w34 = w34DataProcessing("./data_processing_scripts/raw_data_dump/w34/")
    w34.scada_processing(scada_save_folder)
    w34.generate_turbine_lvl_files(turbine_save_folder)
    print("w34 data processing finished")
        
    
