import os
import sys
from datetime import datetime, timedelta
import numpy as np
import metview as mv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.verif_scores import (reliability_diagram_bs, 
                                                      contingency_table_probabilistic_bs, 
                                                      frequency_bias, 
                                                      hit_rate, 
                                                      false_alarm_rate, 
                                                      aroc_trapezium
                                                      )

##############################################################################################################
# CODE DESCRIPTION
# 08_prob_ff_meteo_verif_short_fc.py verifies the performance of short-range rainfall-based predictions of areas at risk of flash floods. 
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)

# Usage: python3 08_prob_ff_meteo_verif_short_fc.py

# Runtime: ~ 4 hours per return period.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# date_s (date, in YYYYMMDD format): start date to consider in the verification period.
# date_f (date, in YYYYMMDD format): fina date to consider in the verification period.
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
# num_bs (integer): number of repetitions in the bootstrapping with replacement.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in_ff (string): relative path of the directory containing the flash flood observations. 
# dir_in_prob_tp (string): relative path of the directory containing the rainfall probabilities of exceeding a certain return period.
# dir_out (string): relative path of the directory containing the values (original and bootstrapped) for the considered verification scores.

##############################################################################################################
# INPUT PARAMETERS
rp = int(sys.argv[1])
date_s = datetime(2021,1,1,0)
date_f = datetime(2024,1,31,0)
num_bs = 1000
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in_ff = "data/processed/03_grid_acc_reports_ff"
dir_in_prob_tp = "data/processed/04_tp_prob_exceed_rp_short_fc"
dir_out = "data/processed/08_prob_ff_meteo_verif_short_fc_test"
##############################################################################################################


# Set the rainfall accumulation period and the number of ensemble members in the considered nwp 
acc = 24
num_em = 99

# Reading the domain's mask
mask = mv.values(mv.read(git_repo + "/" + file_in_mask))
ind_mask = np.where(mask == 1)[0]

# Initialise the variables needed for the verification of the rainfall-based predictions
fb_all = []
aroc_all = []

# Computing the verification scores 
print(f'\n Consindering a rainfall event exceeding a {rp}-year return period')
prob_tp_all = [] # Initialise the variables containing the rainfall probabilities and the flash flood reports
ff_all = []

# Determine the arrays with the rainfall probabilities and yes- and non-events
the_date_s = date_s
while the_date_s <= date_f:

      the_date_f = the_date_s + timedelta(hours = acc)
      print(f' - Reading short-range ERA5-ecPoint rainfall probabilities and flash flood reports valid for the {acc}-hourly accumulation period ending on {the_date_f.strftime("%Y%m%d")} at {the_date_f.strftime("%H")} UTC')
      
      # Reading the rainfall probabilities
      file_in_prob_tp = f'{git_repo}/{dir_in_prob_tp}/{rp}rp/{the_date_f.strftime("%Y%m")}/prob_exceed_rp_{the_date_f.strftime("%Y%m%d")}_00.grib'
      prob_tp = mv.values(mv.read(file_in_prob_tp))[ind_mask]
      prob_tp_all.append(prob_tp)

      # Reading the flash flood reports for the considered accumulation period
      file_in_ff = f'{git_repo}/{dir_in_ff}/{the_date_f.strftime("%Y")}/grid_acc_reports_ff_{the_date_f.strftime("%Y%m%d")}_{the_date_f.strftime("%H")}.grib'
      if os.path.exists(file_in_ff) is True:
            ff = mv.values(mv.read(file_in_ff)>0)[ind_mask] # we assign the value 1 to those grid-boxes with at least on flash floor report in it; 0 otherwise.
      else:
            ff = mv.values(mv.read(file_in_prob_tp)*0)[ind_mask] # no flash floods reported in the considered accumulation period
      ff_all.append(ff)
      
      the_date_s = the_date_s + timedelta(days = 1)

prob_tp_all = np.array(prob_tp_all)
ff_all = np.array(ff_all)

# Verification-based optimisation of best forecast
print(f'\nComputing the bootstrapped reliability diagram')
mean_prob_fc, mean_freq_obs, sharpness = reliability_diagram_bs(ff_all, prob_tp_all, num_bs)

print(f'\nComputing the bootstrapped probabilistic contingency table for {num_bs} bootstraps with replacement') 
h, fa, m, cn = contingency_table_probabilistic_bs(ff_all, prob_tp_all, num_em, num_bs)

print(f'\nComputing the bootstraped frequency bias')
fb = frequency_bias(h, fa, m)

print(f'\nComputing the bootstrapped hit rates and false alarm rates')
hr = hit_rate(h,m)
far = false_alarm_rate(fa, cn)

print(f'\nComputing the bootstrapped area under the ROC curve')
aroc = aroc_trapezium(hr, far)

# Saving the scores
dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp'
os.makedirs(dir_out_temp, exist_ok=True)
np.save(f'{dir_out_temp}/mean_prob_fc.npy', mean_prob_fc)
np.save(f'{dir_out_temp}/mean_freq_obs.npy', mean_freq_obs)
np.save(f'{dir_out_temp}/sharpness.npy', sharpness)
np.save(f'{dir_out_temp}/fb.npy', fb)
np.save(f'{dir_out_temp}/hr.npy', hr)
np.save(f'{dir_out_temp}/far.npy', far)
np.save(f'{dir_out_temp}/aroc.npy', aroc)