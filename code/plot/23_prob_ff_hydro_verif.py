import os
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################################################################################
# CODE DESCRIPTION
# 23_prob_ff_hydro_verif.py plots the verification results for the short- and long-range forecasts.
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall reliability)
#     - recall (overall ability to predict rare events)
#     - f1 (overall ability to predict rare events)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
# Note: It would have been more efficient to vectorise all the computations but, due to memory issues, it was not possible. Otherwise, 
# for the considered domain, we would have not been able to compute more than 100 bootstraps, a number that is well below the 
# recommended standars of at least 1000 repetitions. 

# Usage: python3 23_prob_ff_hydro_verif.py

# Runtime: ~ 1 minute.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# step_f_start (integer, in hours): first final step of the accumulation period to consider. 
# step_f_final (integer, in hours): last final step of the accumulation period to consider. 
# linestyles (list of strings): line styles to plot the scores at different lead times.
# git_repo (string): repository's local path.
# dir_in_short_fc (string): relative path of the directory containing the verification results for the short-range forecasts.
# dir_in_long_fc (string): relative path of the directory containing the verification results for the long-range forecasts.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

##########################################################################################################################################################
# INPUT PARAMETERS
step_f_start = 24
step_f_final = 120
linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1))]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_short_fc = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold_new/gradient_boosting_xgboost"
dir_in_long_fc = "data/processed/19_prob_ff_hydro_long_fc_verif"
dir_out = "data/plot/23_prob_ff_hydro_verif"
##########################################################################################################################################################


# Creating the output directory
dir_in_short_fc_temp = f'{git_repo}/{dir_in_short_fc}'
dir_in_long_fc_temp = f'{git_repo}/{dir_in_long_fc}'
dir_out_temp = f'{git_repo}/{dir_out}'
os.makedirs(dir_out_temp, exist_ok=True)


# Plotting the frequency bias values
plt.figure(figsize=(5, 4))
fb_short_fc = np.load(f'{dir_in_short_fc_temp}/test_scores_test.npy')[3]
fb_long_fc = np.load(f'{dir_in_long_fc_temp}/fb.npy')
fb = np.insert(fb_long_fc, 0, fb_short_fc)
plt.plot(np.arange(0, len(fb)), fb, color="mediumblue", lw = 2)
plt.title("Frequency bias", fontweight='bold', color="#333333", fontsize=14)
plt.xlabel("Lead time [Days]", color = "#333333", fontsize = 12)
plt.ylabel("Frequency bias", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(fb_long_fc) + 1))
plt.xlim([-0.1, len(fb_long_fc) + 0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/fb.png', dpi=1000)
plt.close


# Plotting the recall values
plt.figure(figsize=(5, 4))
recall_short_fc = np.load(f'{dir_in_short_fc_temp}/test_scores_test.npy')[0]
recall_long_fc = np.load(f'{dir_in_long_fc_temp}/recall.npy')
recall = np.insert(recall_long_fc, 0, recall_short_fc)
plt.plot(np.arange(0, len(recall)), recall, color="mediumblue", lw = 2)
plt.title("Recall", fontweight='bold', color="#333333", fontsize=14)
plt.xlabel("Lead time [Days]", color = "#333333", fontsize = 12)
plt.ylabel("F1-score", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(recall_long_fc) + 1))
plt.xlim([-0.1, len(recall_long_fc) + 0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/recall.png', dpi=1000)
plt.close


# Plotting the f1-score values
plt.figure(figsize=(5, 4))
f1_short_fc = np.load(f'{dir_in_short_fc_temp}/test_scores_test.npy')[1]
f1_long_fc = np.load(f'{dir_in_long_fc_temp}/f1.npy')
f1 = np.insert(f1_long_fc, 0, f1_short_fc)
plt.plot(np.arange(0, len(f1)), f1, color="mediumblue", lw = 2)
plt.title("F1-score", fontweight='bold', color="#333333", fontsize=14)
plt.xlabel("Lead time [Days]", color = "#333333", fontsize = 12)
plt.ylabel("F1-score", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(f1_long_fc) + 1))
plt.xlim([-0.1, len(f1_long_fc) + 0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/f1.png', dpi=1000)
plt.close


# Plotting the aroc values
plt.figure(figsize=(5, 4))
aroc_short_fc = np.load(f'{dir_in_short_fc_temp}/test_scores_test.npy')[2]
aroc_long_fc = np.load(f'{dir_in_long_fc_temp}/aroc.npy')
aroc = np.insert(aroc_long_fc, 0, aroc_short_fc)
plt.plot(np.arange(0, len(aroc)), aroc, color="mediumblue", lw = 2)
plt.plot([-0.1, len(aroc_long_fc) + 0.1], [0.5, 0.5], "--", color = "#333333")
plt.title("Area Under the ROC curve", fontweight='bold', color="#333333", fontsize=14)
plt.xlabel("Lead time [Days]", color = "#333333", fontsize = 12)
plt.ylabel("AROC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(aroc_long_fc) + 1))
plt.xlim([-0.1, len(aroc_long_fc) + 0.1])
plt.ylim([0.45, 1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
plt.close


# Plotting the roc curves
plt.figure(figsize=(6, 6))
hr = np.load(f'{dir_in_short_fc_temp}/hr_test.npy')
far = np.load(f'{dir_in_short_fc_temp}/far_test.npy')
plt.plot(far, hr, color="mediumblue", lw = 3, label="reanalysis")

hr = np.load(f'{dir_in_long_fc_temp}/hr.npy', allow_pickle=True)
far = np.load(f'{dir_in_long_fc_temp}/far.npy', allow_pickle=True)
for i, step_f in enumerate(range(step_f_start, step_f_final + 1, 24)):
      plt.plot(far[i], hr[i], linestyle=linestyles[i], color="mediumblue", lw = 1, label = f"t+{step_f}")
plt.plot([0,1], [0,1], color="#333333", lw = 0.5)
plt.title("ROC curve", fontweight='bold', color="#333333", fontsize=14, pad = 25)
plt.xlabel("False alarm rate", color = "#333333", fontsize = 12)
plt.ylabel("Hit rate", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend(loc='lower right', labelcolor='#333333', fontsize=10, ncol=2)
plt.grid(linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/roc_curves.png', dpi=1000)
plt.close


# Plotting the reliability diagrams
plt.figure(figsize=(5, 5))
fc_pred_test = np.load(f'{dir_in_short_fc_temp}/fc_pred_test.npy')
obs_freq_test = np.load(f'{dir_in_short_fc_temp}/obs_freq_test.npy')
plt.plot(fc_pred_test, obs_freq_test, color="mediumblue", lw = 3)
plt.plot([0,1], [0,1], color="#333333", lw = 0.5)
plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
plt.tick_params(axis='x', colors='#333333', labelsize=10)
plt.tick_params(axis='y', colors='#333333', labelsize=10)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.grid(linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/reliability_diagram_short_fc.png', dpi=1000)
plt.close()

fc_pred = np.load(f'{dir_in_long_fc_temp}/fc_pred.npy', allow_pickle=True)
obs_freq = np.load(f'{dir_in_long_fc_temp}/obs_freq.npy', allow_pickle=True)
for i, step_f in enumerate(range(step_f_start, step_f_final + 1, 24)):
      plt.figure(figsize=(5, 5))
      plt.plot(fc_pred[i], obs_freq[i], linestyle=linestyles[i], color="mediumblue", lw = 1)
      plt.plot([0,1], [0,1], color="#333333", lw = 0.5)
      plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
      plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
      plt.tick_params(axis='x', colors='#333333', labelsize=10)
      plt.tick_params(axis='y', colors='#333333', labelsize=10)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.grid(linewidth=0.5, color='gainsboro')
      plt.tight_layout()
      plt.savefig(f'{dir_out_temp}/reliability_diagram_{step_f:03d}.png', dpi=1000)
      plt.close()