import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.plots import roc_curve_ci, reliability_diagram_ci

#############################################################################################################
# CODE DESCRIPTION
# 19_prob_ff_meteo_verif_long_fc.py plots the verification scores for the long-range rainfall-based predictions of areas at risk of 
# flash floods. The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)

# Usage: python3 19_prob_ff_meteo_verif_long_fc.py 2024

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
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
step_f_start = 24
step_f_final = 120
step_disc = 24
alpha = 99
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/09_prob_ff_meteo_verif_long_fc"
dir_out = "data/plot/19_prob_ff_meteo_verif_long_fc"
#############################################################################################################


# Plotting the verification scores 
for rp in rp_list:

      for step_f in range(step_f_start, step_f_final + 1, step_disc):

            print(f'\nPlotting the verification scores for the {rp}-return period and step_f = {step_f}')

            # Set main input/output directories
            dir_in_temp = f'{git_repo}/{dir_in}/{rp}rp/{step_f:03d}'
            dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp/{step_f:03d}'
            os.makedirs(dir_out_temp, exist_ok=True)

            # Plot the roc curve
            hr = np.load(f'{dir_in_temp}/hr.npy')
            far = np.load(f'{dir_in_temp}/far.npy')
            aroc = np.load(f'{dir_in_temp}/aroc.npy')
            fb = np.load(f'{dir_in_temp}/fb.npy')
            file_out = f'{dir_out_temp}/roc.png'
            roc_curve_ci(rp, hr, far, aroc, fb, alpha, file_out)

            # Plot the reliability diagram
            mean_prob_fc =  np.load(f'{dir_in_temp}/mean_prob_fc.npy')
            mean_freq_obs =  np.load(f'{dir_in_temp}/mean_freq_obs.npy') * 100
            sharpness = np.load(f'{dir_in_temp}/sharpness.npy')
            file_out = f'{dir_out_temp}/reliability_diagram.png'
            reliability_diagram_ci(rp, mean_prob_fc, mean_freq_obs, sharpness, alpha, file_out)