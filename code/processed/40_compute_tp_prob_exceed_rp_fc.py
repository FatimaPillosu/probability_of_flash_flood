import os
import sys
from datetime import datetime, timedelta
import numpy as np
import metview as mv

##########################################################################################
# CODE DESCRIPTION
# 40_compute_tp_prob_exceed_rp_fc.py computes the forecast probabilities of 24-hourly rainfall (from 
# ERA5-ecPoint) of exceeding a considered list of return periods.
# Runtime: the code takes up to 24 hours per year.

# INPUT PARAMETERS DESCRIPTION
# year (integer, in YYYY format): year to consider.
# step_f_start (integer, in hours): first final-step of the accumulation period.
# step_f_final (integer, in hours): last final-step of the accumulation period.
# disc_step (integer, in hours): step discretisation.
# rp_list (list of integers): list of the return periods.
# git_repo (string): repository's local path.
# dir_in_climate (string): relative path of the directory containing the 24-hourly rainfall climatology.
# dir_in_fc (string): relative path of the directory containing the 24-hourly rainfall forecasts from ERA5-ecPoint.
# dir_out (string): relative path of the directory containing the forecasts probabilities.

# INPUT PARAMETERS
year = int(sys.argv[1])
step_f_start = 24
step_f_final = 120
disc_step = 24
rp_list = [1, 50]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in_climate = "data/raw/reanalysis/era5_ecpoint/tp_24h_climate_1991_2020"
dir_in_fc = "data/raw/reanalysis/era5_ecpoint/tp_fc_24h"
dir_out = "data/compute/40_tp_prob_exceed_rp_fc"
##########################################################################################


# Defining the period and the accumulation period to consider
base_date_s = datetime(year,1,1,0)
base_date_f = datetime(year,12,31,0)
acc = 24

# Reading the rainfall climatology and the return periods
climate = mv.read(git_repo + "/" + dir_in_climate + "/climate_rp.grib")
rp_computed = np.load(git_repo + "/" + dir_in_climate + "/rp.npy")

# Computing the forecast probabilities for 24-hourly rainfall
base_date = base_date_s
while base_date <= base_date_f:

      for step_f in range(step_f_start, step_f_final + 1, disc_step):

            step_s = step_f - acc

            print(f'\nReading point-rainfall forecasts for {base_date.strftime("%Y%m%d")} at {base_date.strftime("%H")} UTC (t+{step_s}, t+{step_f})')
            tp = mv.read(git_repo + "/" + dir_in_fc + "/" + base_date.strftime("%Y%m") + "/Pt_BC_PERC_" + base_date.strftime("%Y%m%d") + "_" + f"{step_f:03d}" + ".grib2")
            
            for rp in rp_list:

                  ind_climate_rp = np.where(rp_computed == rp)[0]
                  climate_rp = climate[ind_climate_rp]
                  prob = mv.sum( ( tp >= mv.duplicate(climate_rp, int(mv.count(tp))) ) ) / mv.count(tp) * 100
  
                  # Saving the probabilities
                  dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp/{base_date.strftime("%Y%m")}'
                  if not os.path.exists(dir_out_temp):
                        os.makedirs(dir_out_temp)
                  file_out = dir_out_temp + "/tp_prob_exceed_rp_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
                  mv.write(file_out, prob)

      base_date = base_date + timedelta(days=1)