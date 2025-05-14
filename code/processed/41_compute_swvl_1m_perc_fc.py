import os
import sys
from datetime import datetime, timedelta
import metview as mv

###########################################################################################################
# CODE DESCRIPTION
# 41_compute_swvl_1m_perc_fc.py computes the forecasts of  instantaneous percentage of ERA5's soil saturation (swvl) for the 
# top 1m level.
# Runtime: the code takes up to 15 minutes to run.

# INPUT PARAMETERS DESCRIPTION
# year (integer, in YYYY format): year to consider.
# step_f_start (integer, in hours): first final-step of the accumulation period.
# step_f_final (integer, in hours): last final-step of the accumulation period.
# disc_step (integer, in hours): step discretisation.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the volumetric soil water for levels 1 (0-7cm), 2(7-28cm), and 3(28-100cm).
# dir_out (string): relative path of the directory containing the instantaneous percentage of soil saturation.

# INPUT PARAMETERS
year = int(sys.argv[1])
step_f_start = 24
step_f_final = 120
disc_step = 24
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in = "data/raw/reanalysis/era5/swvl_fc"
dir_out = "data/compute/41_swvl_1m_perc_fc"
###########################################################################################################


# Defining the period and the accumulation period to consider
base_date_s = datetime(year,1,1,0)
base_date_f = datetime(year,12,31,0)
acc = 24

# Retrieving the soil type
print("Retrieving the soil type ...")
soil_type = mv.retrieve(
    {"class" : "ea",
     "stream" : "oper", 
     "type" : "an", 
     "expver" : "1", 
     "levtype" : "sfc",
     "param" : "43.128",
     "date": "1940-01-01",
     "time": "00:00:00"
    })
soil_type = mv.bitmap(soil_type, 0) # sostitute the zeros for the sea with missing values to avoid dividing by zero

# Calculating the fields of maximum saturation, field capacity and permanent wilting point using the new soil hydrology scheme (obtained from: https://confluence.ecmwf.int/pages/viewpage.action?pageId=121839768)
soil_type_codes = [1, 2, 3, 4, 5, 6, 7]
pwp = [0.059, 0.151, 0.133, 0.279, 0.335, 0.267, 0.151] # permanent wilting point
fc = [0.242, 0.346, 0.382, 0.448, 0.541, 0.662, 0.346] # field capacity
sat = [0.403, 0.439, 0.430, 0.520, 0.614, 0.766, 0.472] # maximum saturation
pwp_field = ( (soil_type == soil_type_codes[0]) * pwp[0] ) + ( (soil_type == soil_type_codes[1]) * pwp[1] ) + ( (soil_type == soil_type_codes[2]) * pwp[2] ) + ( (soil_type == soil_type_codes[3]) * pwp[3] ) + ( (soil_type == soil_type_codes[4]) * pwp[4] ) + ( (soil_type == soil_type_codes[5]) * pwp[5] ) + ( (soil_type == soil_type_codes[6]) * pwp[6] )
fc_field = ( (soil_type == soil_type_codes[0]) * fc[0] ) + ( (soil_type == soil_type_codes[1]) * fc[1] ) + ( (soil_type == soil_type_codes[2]) * fc[2] ) + ( (soil_type == soil_type_codes[3]) * fc[3] ) + ( (soil_type == soil_type_codes[4]) * fc[4] ) + ( (soil_type == soil_type_codes[5]) * fc[5] ) + ( (soil_type == soil_type_codes[6]) * fc[6] )
sat_field = ( (soil_type == soil_type_codes[0]) * sat[0] ) + ( (soil_type == soil_type_codes[1]) * sat[1] ) + ( (soil_type == soil_type_codes[2]) * sat[2] ) + ( (soil_type == soil_type_codes[3]) * sat[3] ) + ( (soil_type == soil_type_codes[4]) * sat[4] ) + ( (soil_type == soil_type_codes[5]) * sat[5] ) + ( (soil_type == soil_type_codes[6]) * sat[6] )

# Computing the levels of moisture content in the soil
print("Computing the instantaneous percentage to soil saturation:")
base_date = base_date_s
while base_date <= base_date_f:

      for step_f in range(step_f_start, step_f_final + 1, disc_step):

            step_s = step_f - acc

            print(f'\nReading the volumetric soil water for {base_date.strftime("%Y%m%d")} at {base_date.strftime("%H")} UTC (t+{step_s}, t+{step_f})')
      
            # Read the volumetric soil water for level 1 (0 - 7 cm)
            swvl = "swvl1"
            FileIN_1 = git_repo + "/" + dir_in + "/" + swvl + "/" + base_date.strftime("%Y") + "/" + base_date.strftime("%Y%m%d%H") + "/" + swvl + "_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
            swvl1 = mv.read(FileIN_1)
      
            # Read the volumetric soil water for level 2 (7 - 28 cm)
            swvl = "swvl2"
            FileIN_2 = git_repo + "/" + dir_in + "/" + swvl + "/" + base_date.strftime("%Y") + "/" + base_date.strftime("%Y%m%d%H") + "/" + swvl + "_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
            swvl2 = mv.read(FileIN_2)
            
            # Read the volumetric soil water for level 3 (28 - 100 cm)
            swvl = "swvl3"
            FileIN_3 = git_repo + "/" + dir_in + "/" + swvl + "/" + base_date.strftime("%Y") + "/" + base_date.strftime("%Y%m%d%H") + "/" + swvl + "_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}"+ ".grib"
            swvl3 = mv.read(FileIN_3)
            
            # Integrating the volumetric soil water for the top 1m
            swvl = (swvl1*(7-0)/100) + (swvl2*(28-7)/100) + (swvl3*(100-28)/100)
            
            # Defining the water content in the soil (in percentage)
            swvl_perc = swvl / sat_field
            swvl_perc = ((swvl_perc >= 1) * 1) + ((swvl_perc < 1) * swvl_perc) # to correct the few spurious grid-boxes with values of soil moiture >= 1
            
            # Save the field
            dir_out_temp = git_repo + "/" + dir_out + "/" + base_date.strftime("%Y%m")
            if not os.path.exists(dir_out_temp):
                  os.makedirs(dir_out_temp)
            file_out = dir_out_temp + "/swvl_1m_perc_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
            mv.write(file_out, swvl_perc)
      
      base_date = base_date + timedelta(hours=24)