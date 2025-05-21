import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#############################################################################################################
# CODE DESCRIPTION
# 18_prob_ff_meteo_verif_short_fc.py plots the verification scores for the short-range rainfall-based predictions of areas at risk of 
# flash floods. The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)

# Usage: python3 18_prob_ff_meteo_plot_verif_short_fc.py 2024

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
# alpha (integer, from 0 to 100); level of confidence for the confidence intervals. 
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the values (original and bootstrapped) for the considered verification scores.
# dir_out (string): relative path of the directory containing the plots of the verification scores, including confidence intervals.

#############################################################################################################
# INPUT PARAMETERS
rp_list = [1, 5, 10, 20, 50, 100]
alpha = 99
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/08_prob_ff_meteo_verif_short_fc"
dir_out = "data/plot/18_prob_ff_meteo_verif_short_fc"
#############################################################################################################


def roc_curve_ci(hr, far, aroc, fb, file_out):

      # Read the hit rate, false alarm rate, and area under the roc curve for the original dataset
      hr_real = hr[0,:]
      far_real = far[0,:]
      aroc_real = aroc[0]
      fb_real = fb[0,:]

      # Read the hit rate, false alarm rate, and area under the roc curve for the bootstrapped dataset
      hr_bs = hr[1:,:]
      far_bs = far[1:,:]
      aroc_bs = aroc[1:]

      # Compute confidence intervals for the roc curve and the aroc
      cl_lower = (100 - alpha) / 2
      hr_ci_lower = np.percentile(hr_bs, cl_lower, axis = 0)
      far_ci_lower = np.percentile(far_bs, cl_lower, axis = 0)
      aroc_ci_lower = np.percentile(aroc_bs, cl_lower)

      cl_upper = (100 + alpha) / 2 
      hr_ci_upper = np.percentile(hr_bs, cl_upper, axis = 0)
      far_ci_upper = np.percentile(far_bs, cl_upper, axis = 0)
      aroc_ci_upper = np.percentile(aroc_bs, cl_upper)

      x_poly = np.concatenate([far_ci_upper, far_ci_lower[::-1]])
      y_poly = np.concatenate([hr_ci_upper, hr_ci_lower[::-1]])
      
      # Determine the probability threshold for which fb ~ 1 and
      ind_fb_1 = np.where(fb_real >= 1)[0][-1]
      fb_1 = fb_real[ind_fb_1]
      fb_1_x = far_real[ind_fb_1]
      fb_1_y = hr_real[ind_fb_1]

      # Determine the fb for the smallest probability threshold of ~ 1%
      ind_prob_1 = 1
      fb_prob_1 = fb_real[ind_prob_1]
      fb_prob_1_x = far_real[ind_prob_1]
      fb_prob_1_y = hr_real[ind_prob_1]

      # Create the plot of the roc curve with frequency bias and aroc
      fig, ax = plt.subplots(figsize=(5, 5))
      ax.plot(far_real, hr_real, "-o", markersize=2, color="dodgerblue", lw = 1, label = f"AROC = {aroc_real:.3f}")
      ax.fill(x_poly, y_poly, color='dodgerblue', alpha=0.2, label=f"{alpha}% confidence interval\nAROC ({aroc_ci_lower:.3f} , {aroc_ci_upper:.3f})", edgecolor='none')
      plt.plot(fb_1_x, fb_1_y, "o", markersize=4, color = "crimson")
      ax.text(fb_1_x + 0.02, fb_1_y - 0.02, f"FB = {fb_1:.2f} at\nprob_thr = {ind_fb_1}%", fontsize=8, color='#333333', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
      plt.plot(fb_prob_1_x, fb_prob_1_y, "o", markersize=4, color = "pink")
      ax.text(fb_prob_1_x + 0.02, fb_prob_1_y - 0.07, f"FB = {fb_prob_1:.2f} at\nprob_thr = {ind_prob_1}%", fontsize=8, color='#333333', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
      plt.plot([-0.01,1.01], [-0.01,1.01],  "--", color="darkgrey", lw = 1)
      
      ax.set_title(f"tp > {rp}-year return period", color='#333333', fontweight='bold', pad = 30, fontsize = 10)
      ax.set_xlabel("False Alarm Rate [-]", color='#333333')
      ax.set_ylabel("Hit Rate [-] ", color='#333333')
      legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize = 10)
      for text in legend.get_texts():
            text.set_color('#333333')
      
      plt.xlim([-0.01,1.01])
      plt.ylim([-0.01,1.01])
      ax.spines['bottom'].set_color('darkgrey')
      ax.spines['left'].set_color('darkgrey')
      ax.spines['top'].set_color('darkgrey')
      ax.spines['right'].set_color('darkgrey')
      ax.tick_params(axis='x', colors='#333333') 
      ax.tick_params(axis='y', colors='#333333')  
      ax.grid(True, color='gainsboro', linewidth=0.3)

      # Saving the plot
      plt.savefig(file_out, dpi = 1000)
      plt.close()


def reliability_diagram_ci(prob_fc, freq_obs, sharpness, file_out):

      # Read the probabilities for the forecasts and the observational frequencies for the original dataset
      prob_fc_real = prob_fc[0,1:] #  eliminate the sharpness for the first value equal to probability threshold = 0%
      freq_obs_real = freq_obs[0,1:]
      sharpness_real = sharpness[0,1:]

      # Read the probabilities for the forecasts and the observational frequencies for the bootstrapped dataset
      prob_fc_bs = prob_fc[1:,1:] 
      freq_obs_bs = freq_obs[1:,1:]

      # Compute confidence intervals for the reliability diagram
      cl_lower = (100 - alpha) / 2
      prob_fc_ci_lower = np.percentile(prob_fc_bs, cl_lower, axis = 0)
      freq_obs_ci_lower = np.percentile(freq_obs_bs, cl_lower, axis = 0)

      cl_upper = (100 + alpha) / 2 
      prob_fc_ci_upper = np.percentile(prob_fc_bs, cl_upper, axis = 0)
      freq_obs_ci_upper = np.percentile(freq_obs_bs, cl_upper, axis = 0)

      x_poly = np.concatenate([prob_fc_ci_upper, prob_fc_ci_lower[::-1]])
      y_poly = np.concatenate([freq_obs_ci_upper, freq_obs_ci_lower[::-1]])

      # Determine where sharpness becomes 0
      red = 0.01
      sharpness_red = red * sharpness_real[0]
      sharpness_red_ind = np.where(sharpness_real < sharpness_red )[0][0]
      sharpness_red_x = prob_fc_real[sharpness_red_ind]
      sharpness_red_y = sharpness_real[sharpness_red_ind]

      rel_diag_sharpness_red_ind = np.where(prob_fc_real < sharpness_red_x)[0][-1] + 1
      rel_diag_sharpness_red_x = prob_fc_real[rel_diag_sharpness_red_ind]
      rel_diag_sharpness_red_y = freq_obs_real[rel_diag_sharpness_red_ind]

      # Create the plot for the reliability diagram (main plot)
      fig, ax = plt.subplots(figsize=(5, 5))
      ax.plot(prob_fc_real, freq_obs_real, color="dodgerblue", lw = 1, label = "Reliability diagram")
      ax.fill(x_poly, y_poly, color='dodgerblue', alpha=0.2, edgecolor='none', label = f"{alpha}% confidence interval")
      ax.plot(rel_diag_sharpness_red_x, rel_diag_sharpness_red_y, "o", markersize=4, color = "crimson")
      ax.plot([-1,101], [-1,101],  "--", color="darkgrey", lw = 1)
      
      ax.set_title(f"tp > {rp}-year return period", color='#333333', fontweight='bold', pad = 25, fontsize = 10)
      ax.set_xlabel("Forecast Probability [%]", color='#333333')
      ax.set_ylabel("Observed Frequency [%] ", color='#333333')
      legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False, fontsize = 10)
      for text in legend.get_texts():
            text.set_color('#333333')
      
      plt.xlim([-1,101])
      plt.ylim([-1,101])
      ax.spines['bottom'].set_color('darkgrey')
      ax.spines['left'].set_color('darkgrey')
      ax.spines['top'].set_color('darkgrey')
      ax.spines['right'].set_color('darkgrey')
      ax.tick_params(axis='x', colors='#333333') 
      ax.tick_params(axis='y', colors='#333333')  
      ax.grid(True, color='gainsboro', linewidth=0.3)
     
      # Create the plot for sharpness (inset plot)
      inset_ax = fig.add_axes([0.22, 0.63, 0.36, 0.20])
      inset_ax.plot(np.arange(len(sharpness_real)), sharpness_real, color="dodgerblue", lw = 1)
      inset_ax.plot(sharpness_red_x, sharpness_red_y, "o", markersize=3, color = "crimson", label = f"Probability threshold\nat which sharpness\nreduces to {sharpness_red_y}\n(={int(red * 100)}% of the first value)")
      inset_ax.text(sharpness_red_x - 3, sharpness_red_y + 7500, f"{int(sharpness_red_x)}%", fontsize=8, color='#333333')
      
      inset_ax.set_title("Sharpness", fontsize=8, fontweight='bold', pad=3)
      inset_ax.set_xlabel("Forecast Probability [%]", fontsize=8, labelpad=2)
      inset_ax.set_ylabel("Absolute Frequency", fontsize=8, labelpad=1) 
      legend = inset_ax.legend(fontsize = 8, loc = "upper center")
      legend.get_frame().set_linewidth(0) 
      legend.get_frame().set_edgecolor('none')
      
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_scientific(True)
      formatter.set_powerlimits((0, 3))
      inset_ax.yaxis.set_major_formatter(formatter)
      inset_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
      inset_ax.tick_params(axis='both', labelsize=8)
      inset_ax.yaxis.get_offset_text().set_fontsize(8)
      inset_ax.set_xlim(-1, 101)

      inset_ax.spines['bottom'].set_color('darkgrey')
      inset_ax.spines['left'].set_color('darkgrey')
      inset_ax.spines['top'].set_color('darkgrey')
      inset_ax.spines['right'].set_color('darkgrey')
      inset_ax.tick_params(axis='x', colors='#333333') 
      inset_ax.tick_params(axis='y', colors='#333333')  

      # Saving the plot
      plt.savefig(file_out, dpi = 1000)
      plt.close()

####################################################################################################################


# Plotting the verification scores 
for rp in rp_list:

      print(f'\nPlotting the verification scores for the {rp}-return period')

      # Set main input/output directories
      dir_in_temp = f'{git_repo}/{dir_in}/{rp}rp'
      dir_out_temp = f'{git_repo}/{dir_out}/{rp}rp'
      os.makedirs(dir_out_temp, exist_ok=True)

      # Plot the roc curve
      hr = np.load(f'{dir_in_temp}/hr.npy')
      far = np.load(f'{dir_in_temp}/far.npy')
      aroc = np.load(f'{dir_in_temp}/aroc.npy')
      fb = np.load(f'{dir_in_temp}/fb.npy')
      file_out = f'{dir_out_temp}/roc.png'
      roc_curve_ci(hr, far, aroc, fb, file_out)

      # Plot the reliability diagram
      mean_prob_fc =  np.load(f'{dir_in_temp}/mean_prob_fc.npy')
      mean_freq_obs =  np.load(f'{dir_in_temp}/mean_freq_obs.npy')
      sharpness = np.load(f'{dir_in_temp}/sharpness.npy')
      file_out = f'{dir_out_temp}/reliability_diagram.png'
      reliability_diagram_ci(mean_prob_fc, mean_freq_obs, sharpness, file_out)