import os
import sys
from datetime import datetime, timedelta
import numpy as np
import metview as mv

#########################################################################################
# CODE DESCRIPTION
# 05_tp_prob_exceed_rp_long_fc.py computes the probabilities of 24-hourly long-range rainfall (from 
# ERA5-ecPoint) of exceeding a considered list of return periods.

# Usage: python3 05_tp_prob_exceed_rp_long_fc.py 2024

# Runtime: ~ 24 hours per year.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year (integer, in YYYY format): year to consider.
# step_s_start (integer, hours): first start step to consider for the accumulation period.
# step_s_final (integer, hours): final start step to consider for the accumulation period.
# step_disc (integer, hours): discretisation to consider for step_s.
# rp_list (list of integers): list of the return periods.
# git_repo (string): repository's local path.
# dir_in_climate (string): relative path of the directory containing the 24-hourly rainfall climatology.
# dir_in_reanalysis (string): relative path of the directory containing the 24-hourly rainfall from ERA5-ecPoint.
# dir_out (string): relative path of the directory containing the probabilities.

#########################################################################################
# INPUT PARAMETERS
year = int(sys.argv[1])
step_s_start = 0
step_s_final = 96
step_disc = 24
rp_list = [1, 2, 5, 10, 20, 50, 100]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_climate = "data/raw/reanalysis/era5_ecpoint/tp_24h_climate_1991_2020"
dir_in_reanalysis = "data/raw/reanalysis/era5_ecpoint/tp_24h_long_fc"
dir_out = "data/processed/05_tp_prob_exceed_rp_long_fc"
#########################################################################################


# Defining the period and the accumulation period to consider
the_date_s = datetime(year,1,1,0)
the_date_f = datetime(year,12,31,0)
acc = 24

# Reading the rainfall climatology and the return periods
climate = mv.read(git_repo + "/" + dir_in_climate + "/climate_rp.grib")
rp_computed = np.load(git_repo + "/" + dir_in_climate + "/rp.npy")

# Computing the probabilities of 24-hourly long-range rainfall from ERA5-ecPoint exceeding a series of return periods
print()
print("Computing the probabilities of 24-hourly long-range rainfall from ERA5-ecPoint exceeding:")
for rp in rp_list:

      # Selecting the corresponding climatology to the considered return period
      ind_climate_rp = np.where(rp_computed == rp)[0]
      climate_rp = climate[ind_climate_rp]
      climate_rp = mv.duplicate(climate_rp, 99)

      # Computing the probabilities of 24-hourly long-range rainfall from ERA5-ecPoint for a specific step
      for step_s in range(step_s_start, step_s_final+1, step_disc):

            step_f = step_s + acc

            # Computing the probabilities
            the_date = the_date_s
            while the_date <= the_date_f:
                  
                  print(" - " + str(rp) + "-year return period, for forecast on " + the_date.strftime("%Y-%m-%d")  + " at " + the_date.strftime("%H") + " UTC (t+" + str(step_s) + ", t+" + str(step_f) + ")") 

                  tp = mv.read(git_repo + "/" + dir_in_reanalysis + "/" + the_date.strftime("%Y%m") + "/Pt_BC_PERC_" + the_date.strftime("%Y%m%d") + "_" + f"{step_f:03d}" + ".grib2")
                  prob = mv.sum( ( tp >= climate_rp ) ) / mv.count(tp) * 100
      
                  # Saving the probabilities
                  dir_out_temp = git_repo + "/" + dir_out + "/" + str(rp) + "rp" + "/" + the_date.strftime("%Y%m")
                  if not os.path.exists(dir_out_temp):
                        os.makedirs(dir_out_temp)
                  file_out = dir_out_temp + "/prob_exceed_rp_" + the_date.strftime("%Y%m%d") + "_" + the_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
                  mv.write(file_out, prob)

                  the_date = the_date + timedelta(days=1)