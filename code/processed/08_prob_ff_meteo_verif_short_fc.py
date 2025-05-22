import os
import sys
from datetime import datetime, timedelta
import numpy as np
import metview as mv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.verif_scores import (reliability_diagram, 
                                                      contingency_table_probabilistic, 
                                                      frequency_bias, 
                                                      hit_rate, 
                                                      false_alarm_rate, 
                                                      aroc_trapezium
                                                      )
from sklearn.metrics import roc_curve, roc_auc_score

##############################################################################################################
# CODE DESCRIPTION
# 08_prob_ff_meteo_verif_short_fc.py verifies the performance of short-range rainfall-based predictions of areas at risk of flash floods. 
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
# Note: It would have been more efficient to vectorise all the computations but, due to memory issues, it was not possible. Otherwise, 
# for the considered domain, we would have not been able to compute more than 100 bootstraps, a number that is well below the 
# recommended standars of at least 1000 repetitions. 

# Usage: python3 08_prob_ff_meteo_verif_short_fc.py

# Runtime: ~ 1.5 hours.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# rp (integer): rainfall threshold expressed as return periods (in years).
# date_s (date, in YYYYMMDD format): start date to consider in the verification period.
# date_f (date, in YYYYMMDD format): fina date to consider in the verification period.
# num_bs (integer): number of repetitions in the bootstrapping with replacement.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in_ff (string): relative path of the directory containing the flash flood observations. 
# dir_in_prob (string): relative path of the directory containing the rainfall probabilities of exceeding a certain return period.
# dir_out (string): relative path of the directory containing the values (original and bootstrapped) for the considered verification scores.

##############################################################################################################
# INPUT PARAMETERS
rp = int(sys.argv[1])
date_s = datetime(2021,1,1,0)
date_f = datetime(2024,12,31,0)
num_bs = 1000
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in_ff = "data/processed/03_grid_acc_reports_ff"
dir_in_prob = "data/processed/04_tp_prob_exceed_rp_short_fc"
dir_out = "data/processed/08_prob_ff_meteo_verif_short_fc"
##############################################################################################################


# Set the rainfall accumulation period and the number of ensemble members in the considered nwp 
acc = 24
num_em = 99


##############################################
# Creating the verification dataset for the original dates #
##############################################

# Reading the domain's mask
mask = mv.values(mv.read(git_repo + "/" + file_in_mask))
ind_mask = np.where(mask == 1)[0]

# Reading the forecasts probabilities and flash flood reports for the original dataset
prob_original = [] # Initialise the variables containing the rainfall probabilities for the verification period
ff_original = [] # Initialise the variables containing the flash flood reports for the verification period
the_date_s = date_s
print(f'Reading short-range ERA5-ecPoint rainfall probabilities and flash flood reports valid for the {acc}-hourly accumulation period ending on:')
while the_date_s <= date_f:

      the_date_f = the_date_s + timedelta(hours = acc)
      print(f' -  {the_date_f.strftime("%Y%m%d")} at {the_date_f.strftime("%H")} UTC, exceeding {rp}-year return period')
      
      # Reading the rainfall probabilities
      file_in_prob = f'{git_repo}/{dir_in_prob}/{rp}rp/{the_date_f.strftime("%Y%m")}/prob_exceed_rp_{the_date_f.strftime("%Y%m%d")}_00.grib'
      prob = mv.values(mv.read(file_in_prob))[ind_mask]
      prob_original.append(prob)

      # Reading the flash flood reports
      file_in_ff = f'{git_repo}/{dir_in_ff}/{the_date_f.strftime("%Y")}/grid_acc_reports_ff_{the_date_f.strftime("%Y%m%d")}_{the_date_f.strftime("%H")}.grib'
      if os.path.exists(file_in_ff) is True:
            ff = mv.values(mv.read(file_in_ff)>0)[ind_mask] # we assign the value 1 to those grid-boxes with at least on flash floor report in it; 0 otherwise.
      else:
            ff = mv.values(mv.read(file_in_prob)*0)[ind_mask] # no flash floods reported in the considered accumulation period
      ff_original.append(ff)
      
      the_date_s = the_date_s + timedelta(days = 1)

prob_original = np.array(prob_original).astype(np.float32) # array converted to 32 bit to reduce memory usage
ff_original = np.array(ff_original).astype(np.float32)



##########################################
# Verification-based optimisation of best forecast #
##########################################

print("Computing the verification scores (reliability diagram, roc curve, area under the roc, and frequency bias) for the boostrapped dataset n.:")
num_dates = ff_original.shape[0]

hr_bs = []
far_bs = []
fb_bs = []
aroc_bs = []
hr_sl_bs = []
far_sl_bs = []
aroc_sl_bs = []
mean_prob_fc_bs = []
mean_freq_obs_bs = []
sharpness_bs = []

for ind_bs in range(num_bs + 1):

      print(f" - {ind_bs}")

      if ind_bs == 0: # original dataset

            prob_bs = prob_original.ravel()
            ff_bs = ff_original.ravel()

      else: # bootstrapped dataset

            indices = np.random.choice(num_dates, size=num_dates, replace = True)
            prob_bs = prob_original[indices, :].ravel()
            ff_bs = ff_original[indices, :].ravel()

      # Computing the reliability diagram
      mean_prob_fc, mean_freq_obs, sharpness = reliability_diagram(ff_bs, prob_bs)
      mean_prob_fc_bs.append(mean_prob_fc)
      mean_freq_obs_bs.append(mean_freq_obs)
      sharpness_bs.append(sharpness)

      # Computing the contingency table
      h, fa, m, cn = contingency_table_probabilistic(ff_bs, prob_bs, num_em)      

      # Computing the frequency bias
      fb_bs.append(frequency_bias(h, fa, m))

      # Computing the roc curves and the area under the roc
      hr = hit_rate(h,m)
      far = false_alarm_rate(fa, cn)
      hr_bs.append( hr )
      far_bs.append( far )
      aroc_bs.append( aroc_trapezium(hr, far) )

      # Compute the scikit-learn roc curve and area under the roc curve
      far_sl, hr_sl, thresholds = roc_curve(ff_bs, prob_bs)
      aroc_sl = roc_auc_score(ff_bs, prob_bs)
      far_sl_bs.append(far_sl)
      hr_sl_bs.append(hr_sl)
      aroc_sl_bs.append(aroc_sl)

# Saving the scores
dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp'
os.makedirs(dir_out_temp, exist_ok=True)
np.save(f'{dir_out_temp}/mean_prob_fc.npy', np.array(mean_prob_fc_bs))
np.save(f'{dir_out_temp}/mean_freq_obs.npy', np.array(mean_freq_obs_bs))
np.save(f'{dir_out_temp}/sharpness.npy', np.array(sharpness_bs))
np.save(f'{dir_out_temp}/fb.npy', np.array(fb_bs))
np.save(f'{dir_out_temp}/hr.npy', np.array(hr_bs))
np.save(f'{dir_out_temp}/far.npy', np.array(far_bs))
np.save(f'{dir_out_temp}/aroc.npy', np.array(aroc_bs))
np.save(f'{dir_out_temp}/hr_sl.npy', np.array(hr_sl_bs, dtype=object))
np.save(f'{dir_out_temp}/far_sl.npy', np.array(far_sl_bs, dtype=object))
np.save(f'{dir_out_temp}/aroc_sl.npy', np.array(aroc_sl_bs))