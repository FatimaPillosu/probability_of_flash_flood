import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.plots import aroc_ci, fb_ci

#############################################################################################################
# CODE DESCRIPTION
# 20_prob_ff_meteo_verif_overall_scores.py plots the overall verification scores (area under the ROC and frequency bias) for short- and long-range rainfall-based predictions of areas at risk of 
# flash floods. The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)

# Usage: python3 20_prob_ff_meteo_verif_overall_scores.py 

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
# rp_colour_list (list of integers): list of colours to associate with each return period.
# step_f_start (integer, hours): first final step to consider for the accumulation period.
# step_f_final (integer, hours): final final step to consider for the accumulation period.
# step_disc (integer, hours): discretisation to consider for step_f.
# alpha (integer, from 0 to 100); level of confidence for the confidence intervals. 
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the values (original and bootstrapped) for the considered verification scores.
# dir_out (string): relative path of the directory containing the plots of the verification scores, including confidence intervals.

#############################################################################################################
# INPUT PARAMETERS
rp_list = [1, 5, 10, 20, 50, 100]
rp_colour_list = ["crimson", "darkviolet", "yellowgreen", "dodgerblue", "mediumblue", "teal"]
step_f_start = 24
step_f_final = 120
step_disc = 24
alpha = 99
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_short = "data/processed/08_prob_ff_meteo_verif_short_fc"
dir_in_long = "data/processed/09_prob_ff_meteo_verif_long_fc"
dir_out = "data/plot/20_prob_ff_meteo_verif_overall_scores"
#############################################################################################################


# Plotting the verification scores 
aroc_all = []
fb_all = []

for rp in rp_list:

      print(f'Reading the aroc values and frequency bias for the {rp}-return period')
      aroc = [np.load(f'{git_repo}/{dir_in_short}/{rp}rp/aroc.npy')]
      fb = [np.load(f'{git_repo}/{dir_in_short}/{rp}rp/fb.npy')]

      for step_f in range(step_f_start, step_f_final + 1, step_disc):
            aroc.append(np.load(f'{git_repo}/{dir_in_long}/{rp}rp/{step_f:03d}/aroc.npy'))
            fb.append(np.load(f'{git_repo}/{dir_in_long}/{rp}rp/{step_f:03d}/fb.npy'))

      aroc_all.append(np.array(aroc))
      fb_all.append(np.array(fb))

aroc_all = np.stack(aroc_all, axis=2)
fb_all = np.stack(fb_all, axis=2)

# Saving the verification plots
dir_out_temp = f'{git_repo}/{dir_out}'
os.makedirs(dir_out_temp, exist_ok=True)
aroc_ci(rp_list, rp_colour_list, aroc_all, alpha, f'{dir_out_temp}/aroc.png')
fb_ci(rp_list, rp_colour_list, fb_all, alpha, f'{dir_out_temp}/fb.png')