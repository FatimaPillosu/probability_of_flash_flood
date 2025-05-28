import os
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################################################################################
# CODE DESCRIPTION
# 23_prob_ff_hydro_short_fc_verif_test.py plots the verification results for the test data, using the retrained model for the best k-fold.
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
# Note: It would have been more efficient to vectorise all the computations but, due to memory issues, it was not possible. Otherwise, 
# for the considered domain, we would have not been able to compute more than 100 bootstraps, a number that is well below the 
# recommended standars of at least 1000 repetitions. 

# Usage: python3 23_prob_ff_hydro_short_fc_verif_test.py

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
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# colours_ml_trained_list (list of strings): list of colours to associate to each trained model.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

##########################################################################################################################################################
# INPUT PARAMETERS
ml_trained_list = ["gradient_boosting_xgboost"]
colours_ml_trained_list = ["mediumblue"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold_new"
dir_out = "data/plot/23_prob_ff_hydro_short_fc_verif_test"
##########################################################################################################################################################


# Creating the output directory
dir_out_temp = f'{git_repo}/{dir_out}'
os.makedirs(dir_out_temp, exist_ok=True)

# # Plotting the recall values
# plt.figure(figsize=(6, 5))
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
#       recall = np.load(f'{dir_in_temp}/test_scores.npy')
#       plt.plot(np.arange(len(recall)), recall, color=colours_ml_trained, lw = 2, label=ml_trained)
# plt.title("Recall", fontweight='bold', color="#333333", pad=60, fontsize=14)
# plt.xlabel("k_outer", color = "#333333", fontsize = 12)
# plt.ylabel("Recall", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12)
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(np.arange(len(recall)))
# plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
# plt.xlim([-0.1, len(recall)-0.9])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/recall.png', dpi=1000)
# plt.close


# # Plotting the f1-score values
# plt.figure(figsize=(6, 5))
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
#       f1 = np.load(f'{dir_in_temp}/f1.npy')
#       plt.plot(np.arange(len(f1)), f1, color=colours_ml_trained, lw = 2, label=ml_trained)
# plt.title("F1-score", fontweight='bold', color="#333333", pad=60, fontsize=14)
# plt.xlabel("k_outer", color = "#333333", fontsize = 12)
# plt.ylabel("F1-score", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12)
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(np.arange(len(f1)))
# plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
# plt.xlim([-0.1, len(f1)-0.9])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/f1.png', dpi=1000)
# plt.close


# # Plotting the aroc values
# plt.figure(figsize=(6, 5))
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
#       aroc = np.load(f'{dir_in_temp}/aroc.npy')
#       plt.plot(np.arange(len(aroc)), aroc, color=colours_ml_trained, lw = 2, label=ml_trained)
# plt.title("Area Under the ROC curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
# plt.xlabel("k_outer", color = "#333333", fontsize = 12)
# plt.ylabel("AROC", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12)
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(np.arange(len(aroc)))
# plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
# plt.xlim([-0.1, len(aroc)-0.9])
# plt.ylim([0.75, 0.9])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
# plt.close


# # Plotting the training times per fold
# plt.figure(figsize=(6, 5))
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
#       fold_time = np.load(f'{dir_in_temp}/fold_times.npy') / 60
#       plt.plot(np.arange(len(fold_time)), fold_time, color=colours_ml_trained, lw = 2, label=ml_trained)
# plt.title("Training times", fontweight='bold', color="#333333", pad=60, fontsize=14)
# plt.xlabel("k_outer", color = "#333333", fontsize = 12)
# plt.ylabel("Times [hours]", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12)
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(np.arange(len(fold_time)))
# plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
# plt.xlim([-0.1, len(fold_time)-0.9])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/fold_times.png', dpi=1000)
# plt.close

# Plotting the roc curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      
      test_scores_train = np.load(f'{dir_in_temp}/test_scores_train.npy')
      recall_train = test_scores_train[0]
      f1_train = test_scores_train[1]
      yes_thr_train = test_scores_train[3]
      print(f"Training dataset - recall: {recall_train:.3f}, f1-score: {f1_train:.3f}, yes-event threshold: {yes_thr_train:.3f}")

      test_scores_test = np.load(f'{dir_in_temp}/test_scores_test.npy')
      recall_test = test_scores_test[0]
      f1_test = test_scores_test[1]
      yes_thr_test = test_scores_test[3]
      print(f"Test dataset - recall: {recall_test:.3f}, f1-score: {f1_test:.3f}, yes-event threshold: {yes_thr_test:.3f}")


# Plotting the roc curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      
      plt.figure(figsize=(6, 6))
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      
      hr_train = np.load(f'{dir_in_temp}/hr_train.npy')
      far_train = np.load(f'{dir_in_temp}/far_train.npy')
      aroc_train = np.load(f'{dir_in_temp}/test_scores_train.npy')[2]
      plt.plot(far_train, hr_train, "-", color=colours_ml_trained, lw = 2, label = f"Train dataset (AROC = {aroc_train:.3f})")

      hr_test = np.load(f'{dir_in_temp}/hr_test.npy')
      far_test = np.load(f'{dir_in_temp}/far_test.npy')
      aroc_test = np.load(f'{dir_in_temp}/test_scores_test.npy')[2]
      plt.plot(far_test, hr_test, "--", color=colours_ml_trained, lw = 2, label = f"Test dataset (AROC = {aroc_test:.3f})")

      plt.plot([0,1], [0,1], color="#333333", lw = 0.5)
      plt.title("ROC curve", fontweight='bold', color="#333333", fontsize=14, pad = 25)
      plt.xlabel("False alarm rate", color = "#333333", fontsize = 12)
      plt.ylabel("Hit rate", color = "#333333", fontsize = 12)
      plt.tick_params(axis='x', colors='#333333', labelsize=12)
      plt.tick_params(axis='y', colors='#333333', labelsize=12)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
      plt.grid(linewidth=0.5, color='gainsboro')
      plt.tight_layout()
      plt.savefig(f'{dir_out_temp}/roc_curve_{ml_trained}.png', dpi=1000)
      plt.close


# Plotting the reliability diagrams
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      plt.figure(figsize=(6, 6))
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      
      prob_fc_train = np.load(f'{dir_in_temp}/fc_pred_train.npy')
      freq_obs_train = np.load(f'{dir_in_temp}/obs_freq_train.npy')
      plt.plot(prob_fc_train, freq_obs_train, "-", color=colours_ml_trained, lw = 1, label = f"Train dataset")
      
      prob_fc_test = np.load(f'{dir_in_temp}/fc_pred_test.npy')
      freq_obs_test = np.load(f'{dir_in_temp}/obs_freq_test.npy')
      plt.plot(prob_fc_test, freq_obs_test, "--", color=colours_ml_trained, lw = 1, label = f"Test dataset")

      plt.plot([0,1], [0,1], color="#333333", lw = 0.5)
      plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
      plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
      plt.tick_params(axis='x', colors='#333333', labelsize=10)
      plt.tick_params(axis='y', colors='#333333', labelsize=10)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
      plt.grid(linewidth=0.5, color='gainsboro')
      plt.tight_layout()
      plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}.png', dpi=1000)
      plt.close()