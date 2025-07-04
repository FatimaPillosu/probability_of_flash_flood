import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.verif_scores import (contingency_table_probabilistic, 
                                                      precision,
                                                      hit_rate, 
                                                      false_alarm_rate,
                                                      reliability_diagram,
                                                      frequency_bias_overall,
                                                      aroc_trapezium
                                                      )


########################################################################################
# CODE DESCRIPTION
# 23_prob_ff_hydro_short_fc_verif.py plots the verification results over the training and verification dataset.
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall score)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
#     - precision-recall curve (breakdown score for imbalanced datasets)
#     - area under the precision-recall curve (overall performance)

# Usage: python3 23_prob_ff_hydro_short_fc_verif.py

# Runtime: ~ 1 minute.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# ml_trained_list (list of strings): names of the models to train. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - feed_forward_keras
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

########################################################################################
# INPUT PARAMETERS
num_bs = 10
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "gradient_boosting_catboost", "gradient_boosting_lightgbm", "random_forest_lightgbm", "feed_forward_keras"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
dir_out = "data/plot/23_prob_ff_hydro_short_fc_verif"
##############################################################################################################


# Creating the verification plots
for loss_func in ["bce", "weighted_bce"]:

      for eval_metric in ["auc", "auprc"]:

            print(f"\nCreating verification plots for loss_fun = {loss_func}, and eval_metric = {eval_metric}")

            # Creating the input/output directory
            dir_in_temp = f'{git_repo}/{dir_in}/{loss_func}/{eval_metric}'
            dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}'
            os.makedirs(dir_out_temp, exist_ok=True)

            # Initialising the variables storing the overall scores
            auprc_train_all = []
            auprc_test_all = []
            aroc_train_all = []
            aroc_test_all = []
            fb_train_all = []
            fb_test_all = []

            # Initialising the variables containing the distribution of forecast probabilities
            fc_prob_all = []
            fc_all = []
            colour_all = []
            fc_prob_max_all = []

            # Computing the verification scores
            for ml_trained in ml_trained_list:

                  print(f" - Plots for {ml_trained}")

                  # Reading the predictions and observations
                  fc_prob_train = np.load(f"{dir_in_temp}/{ml_trained}/fc_train.npy") * 100
                  fc_prob_test = np.load(f"{dir_in_temp}/{ml_trained}/fc_test.npy") * 100
                  obs_train = np.load(f"{dir_in_temp}/{ml_trained}/obs_train.npy")
                  obs_test = np.load(f"{dir_in_temp}/{ml_trained}/obs_test.npy")
                  prob_thr = np.load(f"{dir_in_temp}/{ml_trained}/best_thr.npy") * 100
                  fc_train = fc_prob_train > prob_thr
                  fc_test = fc_prob_test > prob_thr

                  fc_all.append(np.sum(fc_train) / len(fc_train) * 100)
                  fc_all.append(np.sum(fc_test) / len(fc_test) * 100)

                  fc_prob_all.append(fc_prob_train)
                  fc_prob_all.append(fc_prob_test)

                  fc_prob_max_all.append(np.max(fc_prob_train))
                  fc_prob_max_all.append(np.max(fc_prob_test))

                  colour_all.append("#800080")
                  colour_all.append("#00B0F0")


                  # Computing the contingency table
                  h_train, fa_train, m_train, cn_train = contingency_table_probabilistic(obs_train, fc_prob_train, 100)
                  h_test, fa_test, m_test, cn_test = contingency_table_probabilistic(obs_test, fc_prob_test, 100)
                  

                  # Plotting the precision-recall curve
                  p_train = precision(h_train, fa_train)
                  hr_train = hit_rate(h_train, m_train)
                  ref_train = np.sum(obs_train) / len(obs_train)
                  auprc_train_all.append(average_precision_score(obs_train, fc_prob_train))
                  plt.plot(hr_train, p_train, "-o", color = "#800080", lw = 1, ms=2)
                  plt.plot([0,1], [ref_train, ref_train], color = "#333333", lw = 1)

                  p_test = precision(h_test, fa_test)
                  hr_test = hit_rate(h_test, m_test)
                  ref_test = np.sum(obs_test) / len(obs_test)
                  auprc_test_all.append(average_precision_score(obs_test, fc_prob_test))
                  plt.plot(hr_test, p_test, "--o", color = "#00B0F0", lw = 1, ms=2)
                  plt.plot([0,1], [ref_test, ref_test], "--", color = "#333333", lw = 1)

                  plt.xlabel("Recall", color = "#333333", fontsize = 12)
                  plt.ylabel("Precision", color = "#333333", fontsize = 12)
                  plt.tick_params(axis='x', colors='#333333', labelsize=12, )
                  plt.tick_params(axis='y', colors='#333333', labelsize=12)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-0.05,1.05])
                  plt.ylim([-0.05,1.05])
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/pr_curve_{ml_trained}.png', dpi=1000)
                  plt.close()


                  # Plotting the ROC curve - Trapezium and Continuous
                  plt.figure(figsize=(6, 5))

                  hr_train = hit_rate(h_train, m_train)
                  far_train = false_alarm_rate(fa_train, cn_train)
                  aroc_train = aroc_trapezium(hr_train, far_train)
                  plt.plot(far_train, hr_train, "-o", color = "#800080", lw = 2, ms=4, label = f"AROC = {aroc_train:.3f}, min prob exceed= 1%")

                  far_train_c, hr_train_c, thr_roc = roc_curve(obs_train, fc_prob_train)
                  aroc_train_c = auc(far_train_c, hr_train_c)
                  aroc_train_all.append(aroc_train_c)
                  plt.plot(far_train_c, hr_train_c, "--", color = "#800080", lw = 2, ms=2, label = f"AROC_cont = {aroc_train_c:.3f}, min prob exceed={(thr_roc[-1] * 100):.2f}%")
                  
                  hr_test = hit_rate(h_test, m_test)
                  far_test = false_alarm_rate(fa_test, cn_test)
                  aroc_test = aroc_trapezium(hr_test, far_test)
                  plt.plot(far_test, hr_test, "-o", color = "#00B0F0", lw = 2, ms=4, label = f"AROC = {aroc_test:.3f}, min prob exceed= 1%")

                  far_test_c, hr_test_c, thr_roc = roc_curve(obs_test, fc_prob_test)
                  aroc_test_c = auc(far_test_c, hr_test_c)
                  aroc_test_all.append(aroc_test_c)
                  plt.plot(far_test_c, hr_test_c, "--", color = "#00B0F0", lw = 2, ms=2, label = f"AROC_cont = {aroc_test_c:.3f}, min prob exceed={(thr_roc[-1] * 100):.2f}%")
                  
                  plt.plot([0,1], [0, 1], "-", color = "#333333", lw = 1)
                  plt.xlabel("False Alarm Rate", color = "#333333", fontsize = 12)
                  plt.ylabel("Hit Rate", color = "#333333", fontsize = 12)
                  plt.tick_params(axis='x', colors='#333333', labelsize=12, )
                  plt.tick_params(axis='y', colors='#333333', labelsize=12)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-0.05,1.05])
                  plt.ylim([-0.05,1.05])
                  plt.legend()
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/roc_curve_{ml_trained}.png', dpi=1000)
                  plt.close()


                  # Plotting the reliability diagram
                  fig, ax = plt.subplots(figsize=(6, 6))

                  mean_prob_fc_train, mean_freq_obs_train, sharpness_train = reliability_diagram(obs_train, fc_prob_train)
                  plt.plot(mean_prob_fc_train, mean_freq_obs_train * 100, "-o", color = "#800080", lw = 1, ms=2)
                  
                  mean_prob_fc_test, mean_freq_obs_test, sharpness_test = reliability_diagram(obs_test, fc_prob_test)
                  plt.plot(mean_prob_fc_test, mean_freq_obs_test * 100, "-o", color = "#00B0F0", lw = 1, ms=2)
                  
                  plt.plot([0,100], [0, 100], color = "#333333", lw = 1)
                  plt.xlabel("Forecast probability", color = "#333333", fontsize = 12)
                  plt.ylabel("Observation frequency", color = "#333333", fontsize = 12)
                  plt.tick_params(axis='x', colors='#333333', labelsize=12, )
                  plt.tick_params(axis='y', colors='#333333', labelsize=12)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-1,101])
                  plt.ylim([-1,101])
                  plt.tight_layout()

                  inset_ax = fig.add_axes([0.2, 0.7, 0.35, 0.23])
                  inset_ax.plot(np.arange(len(sharpness_train)), sharpness_train, color="#800080", lw = 1)
                  inset_ax.plot(np.arange(len(sharpness_test)), sharpness_test, color="#00B0F0", lw = 1)
                  inset_ax.set_title("Sharpness", fontsize=8, fontweight='bold', pad=3)
                  inset_ax.set_xlabel("Forecast Probability [%]", fontsize=8, labelpad=2)
                  inset_ax.set_ylabel("Absolute Frequency", fontsize=8, labelpad=1) 

                  plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}.png', dpi=1000)
                  plt.close()

                  # Computing the frequency bias
                  fb_train_all.append(frequency_bias_overall(obs_train, fc_prob_train))
                  fb_test_all.append(frequency_bias_overall(obs_test, fc_prob_test))


            # Plotting the overall scores - AROC
            plt.plot(ml_trained_list, aroc_train_all, "-o", color = "#800080", lw = 1, ms=2)
            plt.plot(ml_trained_list, aroc_test_all, "-o", color = "#00B0F0", lw = 1, ms=2)
            plt.ylabel("AROC", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12, )
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(rotation=20)
            plt.grid(axis='y', linewidth=0.5, color='gainsboro')
            plt.ylim([0.5,1])
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
            plt.close()
      
            # Plotting the overall scores - AUPRC
            plt.plot(ml_trained_list, auprc_train_all, "-o", color = "#800080", lw = 1, ms=2)
            plt.plot(ml_trained_list, auprc_test_all, "-o", color = "#00B0F0", lw = 1, ms=2)
            plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12, )
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(rotation=20)
            plt.grid(axis='y', linewidth=0.5, color='gainsboro')
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/auprc.png', dpi=1000)
            plt.close()

            # Plotting the overall scores - FB
            plt.plot(ml_trained_list, fb_train_all, "-o", color = "#800080", lw = 1, ms=2)
            plt.plot(ml_trained_list, fb_test_all, "-o", color = "#00B0F0", lw = 1, ms=2)
            plt.ylabel("FB", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12, )
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(rotation=20)
            plt.grid(axis='y', linewidth=0.5, color='gainsboro')
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/fb.png', dpi=1000)
            plt.close()

            # Creating the distribution plots showing the distribution of forecast probabilities and yes-events
            fig, ax = plt.subplots(figsize=(2, 6))
            plt.plot(fc_prob_max_all, np.arange(len(fc_prob_max_all)), "o")
            plt.xlim([30,105])
            plt.savefig(f'{dir_out_temp}/max_fc_prob.png', dpi=1000)
            plt.close()
            
            fig, ax = plt.subplots(figsize=(5, 6))
            plt.barh(np.arange(len(fc_all)), fc_all, color=colour_all)
            plt.xlim([-0.01, 0.5])
            plt.savefig(f'{dir_out_temp}/yes_events_freq.png', dpi=1000)
            plt.close()
      
            fig, ax = plt.subplots(figsize=(5, 6))
            bp = ax.boxplot(
                        fc_prob_all, 
                        vert=False, 
                        patch_artist=True,
                        showmeans=True,
                        whis=(10, 90),
                        showfliers=False,
                        whiskerprops=dict(color='#333333'),
                        capprops=dict(color='#333333'),
                        medianprops=dict(color='#666666')
                        )
            for patch, color in zip(bp['boxes'], colour_all):
                  patch.set_facecolor(color)
            plt.xlim([-0.1, 2.5])
            plt.savefig(f'{dir_out_temp}/fc_prob_distr.png', dpi=1000)
            plt.close()