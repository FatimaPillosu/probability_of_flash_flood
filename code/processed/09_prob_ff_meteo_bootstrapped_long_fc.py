import os
from datetime import datetime, timedelta
import numpy as np
import metview as mv

####################################################################################################################
# CODE DESCRIPTION
# 09_prob_ff_meteo_bootstrapped_long_fc.py creates the bootstrapped dataset for the rainfall-based long-range forecasts of areas at risk of 
# flash floods.

# Usage: python3 09_prob_ff_meteo_bootstrapped_long_fc.py

# Runtime: ~ 2 hours.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# base_date_s (date, in YYYYMMDDHH format): start base date to consider in the verification period, including forecast run time (given by HH).
# base_date_f (date, in YYYYMMDDHH format): final base date to consider in the verification period, including forecast run time (given by HH).
# step_f_start (integer, in hours): first final step of the considered accumulation period.
# step_f_final (integer, in hours): last final step of the considered accumulation period.
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
# num_bs (integer): number of repetitions in the bootstrapping with replacement.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in_ff (string): relative path of the directory containing the flash flood observations. 
# dir_in_prob (string): relative path of the directory containing the rainfall probabilities of exceeding a certain return period.
# dir_out (string): relative path of the directory containing the values (original and bootstrapped) for the considered verification scores.

####################################################################################################################
# INPUT PARAMETERS
base_date_s = datetime(2021,1,1,0)
base_date_f = datetime(2024,12,31,0)
step_f_start = 24
step_f_final = 120
rp_list = [1,5,10,20,50,100]
num_bs = 100
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in_ff = "data/processed/03_grid_acc_reports_ff"
dir_in_prob = "data/processed/05_tp_prob_exceed_rp_long_fc"
dir_out = "data/processed/09_prob_ff_meteo_bootstrapped_long_fc"
####################################################################################################################


# Set the rainfall accumulation period, the discretisation for the steps, and the number of ensemble members in the considered nwp 
acc = 24
disc_step = 24
num_em = 99

# Reading the domain's mask
mask = mv.values(mv.read(git_repo + "/" + file_in_mask))
ind_mask = np.where(mask == 1)[0]

print(f'Reading long-range ERA5-ecPoint rainfall probabilities and flash flood reports valid for the {acc}-hourly accumulation period:')

# Boostrapping the forecasts and observation datasets for a specific return period
for rp in rp_list:

      # Boostrapping the forecasts and observation datasets for a specific lead time
      for step_f in range(step_f_start, step_f_final + 1, disc_step):
      
            step_s = step_f - acc

            # Reading the forecasts probabilities and flash flood reports for the original dataset
            prob_all = [] # Initialise the variables containing the rainfall probabilities for the verification period
            ff_all = [] # Initialise the variables containing the flash flood reports for the verification period
            base_date = base_date_s

            while base_date <= base_date_f:

                  print(f' -  {base_date.strftime("%Y%m%d")} at {base_date.strftime("%H")} UTC (t+{step_s}, t+{step_f}), exceeding {rp}-year return period')
                  
                  # Reading the rainfall probabilities
                  file_in_prob = f'{git_repo}/{dir_in_prob}/{rp}rp/{base_date.strftime("%Y%m")}/prob_exceed_rp_{base_date.strftime("%Y%m%d")}_{base_date.strftime("%H")}_{step_f:03d}.grib'
                  prob = mv.values(mv.read(file_in_prob))[ind_mask]
                  prob_all.append(prob)

                  # Reading the flash flood reports
                  vt = base_date + timedelta(hours = step_f)
                  file_in_ff = f'{git_repo}/{dir_in_ff}/{vt.strftime("%Y")}/grid_acc_reports_ff_{vt.strftime("%Y%m%d")}_{vt.strftime("%H")}.grib'
                  if os.path.exists(file_in_ff) is True:
                        ff = mv.values(mv.read(file_in_ff)>0)[ind_mask] # we assign the value 1 to those grid-boxes with at least on flash floor report in it; 0 otherwise.
                  else:
                        ff = mv.values(mv.read(file_in_prob)*0)[ind_mask] # no flash floods reported in the considered accumulation period
                  ff_all.append(ff)
                  
                  base_date = base_date + timedelta(days = 1)

            prob_all = np.array(prob_all).astype(np.float32) # array converted to 32 bit to reduce memory usage
            ff_all = np.array(ff_all).astype(np.float32)

            num_dates = ff_all.shape[0]
            indices = np.concatenate([np.arange(num_dates)[:, None], np.random.choice(num_dates, size=(num_bs, num_dates), replace = True).T], axis=1) 
            prob_bs = np.transpose( np.take(prob_all, indices, axis=0), (0, 2, 1) ).reshape(-1, num_bs + 1)
            ff_bs = np.transpose( np.take(ff_all, indices, axis=0), (0, 2, 1) ).reshape(-1, num_bs + 1) * 100

            # Saving the boostrapped datasets
            dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp'
            os.makedirs(dir_out_temp, exist_ok=True)
            np.save(f'{dir_out_temp}/prob_bs_{step_f:03d}.npy', prob_bs)
            np.save(f'{dir_out_temp}/ff_bs_{step_f:03d}.npy', ff_bs)