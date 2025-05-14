import os
import sys
from datetime import datetime, timedelta
import numpy as np
import metview as mv

#########################################################################################
# CODE DESCRIPTION
# 08_compute_tp_prob_exceed_rp.py computes the probabilities of 24-hourly rainfall (from ERA5-ecPoint)
# of exceeding a considered list of return periods.
# Runtime: the code takes up to 24 hours per year.

# INPUT PARAMETERS DESCRIPTION
# year (integer, in YYYY format): year to consider.
# rp_list (list of integers): list of the return periods.
# git_repo (string): repository's local path.
# dir_in_climate (string): relative path of the directory containing the 24-hourly rainfall climatology.
# dir_in_reanalysis (string): relative path of the directory containing the 24-hourly rainfall from ERA5-ecPoint.
# dir_out (string): relative path of the directory containing the probabilities.

# INPUT PARAMETERS
year = int(sys.argv[1])
rp_list = [1, 2, 5, 10, 20, 50, 100]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in_climate = "data/raw/reanalysis/era5_ecpoint/tp_24h_climate_1991_2020"
dir_in_reanalysis = "data/raw/reanalysis/era5_ecpoint/tp_24h"
dir_out = "data/compute/08_tp_prob_exceed_rp"
#########################################################################################


# Defining the period and the accumulation period to consider
the_date_s = datetime(year,1,1)
the_date_f = datetime(year,12,31)
acc = 24

# Reading the rainfall climatology and the return periods
climate = mv.read(git_repo + "/" + dir_in_climate + "/climate_rp.grib")
rp_computed = np.load(git_repo + "/" + dir_in_climate + "/rp.npy")

# Computing the probabilities of 24-hourly rainfall from ERA5-ecPoint exceeding a series of return periods
print()
print("Computing the probabilities of 24-hourly rainfall from ERA5-ecPoint exceeding:")
for rp in rp_list:

      # Selecting the corresponding climatology to the considered return period
      ind_climate_rp = np.where(rp_computed == rp)[0]
      climate_rp = climate[ind_climate_rp]
      climate_rp = mv.duplicate(climate_rp, 99)

      # Computing the probabilities
      the_date = the_date_s
      while the_date <= the_date_f:
            
            the_date_time_final = the_date + timedelta(hours=24)
            print(" - " + str(rp) + "-year return period, ending on " + the_date_time_final.strftime("%Y-%m-%d") + " at " + the_date_time_final.strftime("%H") + " UTC")
            
            tp = mv.read(git_repo + "/" + dir_in_reanalysis + "/" + the_date.strftime("%Y%m") + "/Pt_BC_PERC_" + the_date.strftime("%Y%m%d") + "_024.grib2")
            prob = mv.sum( ( tp >= climate_rp ) ) / mv.count(tp) * 100
  
            # Saving the probabilities
            dir_out_temp = git_repo + "/" + dir_out + "/" + str(rp) + "rp" + "/" + the_date_time_final.strftime("%Y%m")
            if not os.path.exists(dir_out_temp):
                  os.makedirs(dir_out_temp)
            file_out = dir_out_temp + "/prob_exceed_rp_" + the_date_time_final.strftime("%Y%m%d") + "_" + the_date_time_final.strftime("%H") + ".grib"
            mv.write(file_out, prob)

            the_date = the_date + timedelta(days=1)