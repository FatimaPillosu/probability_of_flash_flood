import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

##############################################################################################################
# CODE DESCRIPTION
# 23_prob_ff_hydro_short_fc_verify.py plots the verification results of the training of diverse machine learning models
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
#     - precion-recall curve (breakdown score for imbalanced datasets)

# Usage: python3 23_prob_ff_hydro_short_fc_verify.py

# Runtime: ~ 1 minute.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# num_k_outer (positive integer): number of outer k-folds considered in the nested cross-validation.
# ml_trained_list (list of strings): names of the models to train. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# colours_ml_trained_list (list of strings): list of colours to associate to each trained model.
# eval_metric (string): evaluation metric used for hyperparameter tuning.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

##############################################################################################################
# INPUT PARAMETERS
kfold = 1
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "gradient_boosting_catboost", "gradient_boosting_lightgbm", "random_forest_lightgbm", "gradient_boosting_adaboost", "feed_forward_keras"]
ml_trained_list = ["gradient_boosting_lightgbm"]
colours_ml_trained_list = ["mediumblue", "orangered", "teal", "crimson", "dodgerblue", "darkviolet", "magenta"]
eval_metric = "auprc"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/plot/23_prob_ff_hydro_short_fc_verify"
##############################################################################################################


# Creating the output directory
dir_out_temp = f'{git_repo}/{dir_out}/{eval_metric}'
os.makedirs(dir_out_temp, exist_ok=True)


# # Plotting the aroc values
# plt.figure(figsize=(6, 5))
# score_train = []
# score_test = []
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
#       score_train.append(np.load(f'{dir_in_temp}/aroc_train.npy'))
#       score_test.append(np.load(f'{dir_in_temp}/aroc_test.npy'))
# plt.plot(np.array(ml_trained_list), np.array(score_test), color = "grey", lw = 2, label=ml_trained)
# plt.plot(np.array(ml_trained_list), np.array(score_train), "--", color="grey", lw = 1)
# plt.ylabel("AROC", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12, )
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(rotation=45)
# plt.xlim([-1, len(ml_trained_list) + 1])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
# plt.close


# # Plotting the auprc values
# plt.figure(figsize=(6, 5))
# score_train = []
# score_test = []
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
#       score_train.append(np.load(f'{dir_in_temp}/auprc_train.npy'))
#       score_test.append(np.load(f'{dir_in_temp}/auprc_test.npy'))
# plt.plot(np.array(ml_trained_list), np.array(score_test), color = "grey", lw = 2, label=ml_trained)
# plt.plot(np.array(ml_trained_list), np.array(score_train), "--", color="grey", lw = 1)
# plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12, )
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(rotation=45)
# plt.xlim([-1, len(ml_trained_list) + 1])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/auprc.png', dpi=1000)
# plt.close


# # Plotting the fb values
# plt.figure(figsize=(6, 5))
# score_train = []
# score_test = []
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
#       score_train.append(np.load(f'{dir_in_temp}/fb_train.npy'))
#       score_test.append(np.load(f'{dir_in_temp}/fb_test.npy'))
# plt.plot(np.array(ml_trained_list), np.array(score_test), color = "grey", lw = 2, label=ml_trained)
# plt.plot(np.array(ml_trained_list), np.array(score_train), "--", color="grey", lw = 1)
# plt.ylabel("FB", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12, )
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(rotation=45)
# plt.xlim([-1, len(ml_trained_list) + 1])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/fb.png', dpi=1000)
# plt.close


# # Plotting the best thresholds to convert the probabilistic events to yes-events
# plt.figure(figsize=(6, 5))
# plt.figure(figsize=(6, 5))
# score_train = []
# score_test = []
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
#       score_train.append(np.load(f'{dir_in_temp}/best_thr_train.npy'))
#       score_test.append(np.load(f'{dir_in_temp}/best_thr_test.npy'))
# plt.plot(np.array(ml_trained_list), np.array(score_test), color = "grey", lw = 2, label=ml_trained)
# plt.plot(np.array(ml_trained_list), np.array(score_train), "--", color="grey", lw = 1)
# plt.ylabel("FB", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12, )
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.xticks(rotation=45)
# plt.xlim([-1, len(ml_trained_list) + 1])
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.tight_layout()
# plt.savefig(f'{dir_out_temp}/best_thr.png', dpi=1000)
# plt.close

# Plotting the precision-recall curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      plt.figure(figsize=(5, 5))
      dir_in_temp = f'{git_repo}/{dir_in}/weighted_bce/{eval_metric}/{ml_trained}'
      precision_train = np.load(f'{dir_in_temp}/precision_rep1_fold1.npy')
      #precision_test = np.load(f'{dir_in_temp}/precision_test.npy')
      recall_train = np.load(f'{dir_in_temp}/recall_rep1_fold1.npy')
      #recall_test = np.load(f'{dir_in_temp}/recall_test.npy')
      plt.plot(recall_train, precision_train, "--", color=colours_ml_trained, lw = 0.5)
      #plt.plot(recall_test, precision_test, color=colours_ml_trained, lw = 1)
      plt.plot([0,1], [1,0], color="#333333", lw = 1)
      plt.title("Precision-recall curve", fontweight='bold', color="#333333", fontsize=14)
      plt.xlabel("Recall (also Hit Rate)", color = "#333333", fontsize = 12)
      plt.ylabel("Precision", color = "#333333", fontsize = 12)
      plt.tick_params(axis='x', colors='#333333', labelsize=12)
      plt.tick_params(axis='y', colors='#333333', labelsize=12)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.grid(linewidth=0.5, color='gainsboro')
      plt.tight_layout()
      plt.show()
      exit()
      plt.savefig(f'{dir_out_temp}/precision_recall_curve_{ml_trained}.png', dpi=1000)
      plt.close


# # Plotting the roc curves
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       plt.figure(figsize=(5, 5))
#       dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
#       hr_train = np.load(f'{dir_in_temp}/hr_train.npy')
#       hr_test = np.load(f'{dir_in_temp}/hr_test.npy')
#       far_train = np.load(f'{dir_in_temp}/far_train.npy')
#       far_test = np.load(f'{dir_in_temp}/far_test.npy')
#       plt.plot(far_train, hr_train, "--", color=colours_ml_trained, lw = 1)
#       plt.plot(far_test, hr_test, color=colours_ml_trained, lw = 2)
#       plt.plot([0,1], [0,1], "--", color="#333333", lw = 1)
#       plt.title("ROC curve", fontweight='bold', color="#333333", fontsize=14)
#       plt.xlabel("False alarm rate", color = "#333333", fontsize = 12)
#       plt.ylabel("Hit rate", color = "#333333", fontsize = 12)
#       plt.tick_params(axis='x', colors='#333333', labelsize=12)
#       plt.tick_params(axis='y', colors='#333333', labelsize=12)
#       plt.xlim([-0.05, 1.05])
#       plt.ylim([-0.05, 1.05])
#       plt.grid(linewidth=0.5, color='gainsboro')
#       plt.tight_layout()
#       plt.savefig(f'{dir_out_temp}/roc_curve_{ml_trained}.png', dpi=1000)
#       plt.close


# Plotting the reliability diagrams
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      plt.figure(figsize=(5, 5))
      dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}'
      obs_freq_train = np.load(f'{dir_in_temp}/obs_freq_rep1_fold2.npy')
      #obs_freq_test = np.load(f'{dir_in_temp}/obs_freq_test.npy')
      prob_pred_train = np.load(f'{dir_in_temp}/prob_pred_rep1_fold2.npy')
      #prob_pred_test = np.load(f'{dir_in_temp}/prob_pred_test.npy')
      plt.plot(prob_pred_train, obs_freq_train, "--", color=colours_ml_trained, lw = 1)
      #plt.plot(prob_pred_test, obs_freq_test, color=colours_ml_trained, lw = 2)
      plt.plot([0,1], [0,1], "--", color="#333333", lw = 1)
      plt.title("Reliability diagram", fontweight='bold', color="#333333", fontsize=14)
      plt.xlabel("Forecast probability", color = "#333333", fontsize = 12)
      plt.ylabel("Observation frequency", color = "#333333", fontsize = 12)
      plt.tick_params(axis='x', colors='#333333', labelsize=12)
      plt.tick_params(axis='y', colors='#333333', labelsize=12)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.grid(linewidth=0.5, color='gainsboro')
      plt.tight_layout()
      plt.show()
      exit()

      plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}.png', dpi=1000)
      plt.close