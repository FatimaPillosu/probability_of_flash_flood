import os
import sys
import numpy as np
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
# 11_prob_ff_meteo_verif_long_fc.py verifies the performance of long-range rainfall-based predictions of areas at risk of flash floods. 
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)

# Usage: python3 11_prob_ff_meteo_verif_long_fc.py

# Runtime: ~ 4 hours per return period.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
# step_f_start (integer, hours): first final step to consider.
# step_f_final (integer, hours): last final step to consider.
# step_disc (integer, hours): discretisation to consider for the steps.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the boostrapped values of the flash flood forecasts and observations.
# dir_out (string): relative path of the directory containing the values (original and bootstrapped) for the considered verification scores.

##############################################################################################################
# INPUT PARAMETERS
rp = int(sys.argv[1])
step_f_start = 24
step_f_final = 120
step_disc = 24
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/09_prob_ff_meteo_bootstrapped_long_fc"
dir_out = "data/processed/11_prob_ff_meteo_verif_long_fc"
##############################################################################################################


# Set the number of ensemble members in the considered nwp 
num_em = 99

# Computing the verification scores
for step_f in range(step_f_start, step_f_final + 1, step_disc):

      # Reading the boostrapped forecasts and observations 
      print(f'\n Consindering a rainfall event exceeding a {rp}-year return period')
      prob_bs = np.load(f'{git_repo}/{dir_in}/{rp}rp/prob_bs_{step_f:03d}.npy')
      ff_bs = np.load(f'{git_repo}/{dir_in}/{rp}rp/ff_bs_{step_f:03d}.npy')

      # Verification-based optimisation of best forecast
      print(f'\nComputing the bootstrapped reliability diagram')
      mean_prob_fc, mean_freq_obs, sharpness = reliability_diagram_bs(ff_bs, prob_bs)

      print(f'\nComputing the bootstrapped probabilistic contingency table') 
      h, fa, m, cn = contingency_table_probabilistic_bs(ff_bs, prob_bs, num_em)

      print(f'\nComputing the bootstraped frequency bias')
      fb = frequency_bias(h, fa, m)

      print(f'\nComputing the bootstrapped hit rates and false alarm rates')
      hr = hit_rate(h,m)
      far = false_alarm_rate(fa, cn)

      print(f'\nComputing the bootstrapped area under the ROC curve')
      aroc = aroc_trapezium(hr, far)

      # Saving the scores
      dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp/{step_f:03d}'
      os.makedirs(dir_out_temp, exist_ok=True)
      np.save(f'{dir_out_temp}/mean_prob_fc.npy', mean_prob_fc)
      np.save(f'{dir_out_temp}/mean_freq_obs.npy', mean_freq_obs)
      np.save(f'{dir_out_temp}/sharpness.npy', sharpness)
      np.save(f'{dir_out_temp}/fb.npy', fb)
      np.save(f'{dir_out_temp}/hr.npy', hr)
      np.save(f'{dir_out_temp}/far.npy', far)
      np.save(f'{dir_out_temp}/aroc.npy', aroc)