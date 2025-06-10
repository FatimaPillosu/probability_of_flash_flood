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
num_k_outer = 5
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "gradient_boosting_catboost", "gradient_boosting_lightgbm", "random_forest_lightgbm", "gradient_boosting_adaboost"]
colours_ml_trained_list = ["mediumblue", "orangered", "teal", "crimson", "dodgerblue", "darkviolet", "magenta"]
eval_metric = "auprc"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
file_in_test  = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
dir_out = "data/plot/22_prob_ff_hydro_short_fc_results_train_ml_cv_optuna"
##############################################################################################################


# Creating the output directory
dir_out_temp = f'{git_repo}/{dir_out}/{eval_metric}'
os.makedirs(dir_out_temp, exist_ok=True)


# Plotting the precision-recall curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      plt.figure(figsize=(5, 5))
      for ind_k in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{ind_k + 1}'
            precision_train = np.load(f'{dir_in_temp}/precision_train.npy')
            precision_test = np.load(f'{dir_in_temp}/precision_test.npy')
            recall_train = np.load(f'{dir_in_temp}/recall_train.npy')
            recall_test = np.load(f'{dir_in_temp}/recall_test.npy')
            plt.plot(recall_train, precision_train, "--", color=colours_ml_trained, lw = 0.5)
            plt.plot(recall_test, precision_test, color=colours_ml_trained, lw = 1)
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
      plt.savefig(f'{dir_out_temp}/precision_recall_curve_{ml_trained}.png', dpi=1000)
      plt.close


# Plotting the aroc values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      aroc_train = []
      aroc_test = []
      for kfold in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
            aroc_train.append(np.load(f'{dir_in_temp}/aroc_train.npy'))
            aroc_test.append(np.load(f'{dir_in_temp}/aroc_test.npy'))
      plt.plot(np.arange(1,len(aroc_test) + 1), np.array(aroc_test), color=colours_ml_trained, lw = 2, label=ml_trained)
      plt.plot(np.arange(1,len(aroc_train) + 1), np.array(aroc_train), "--", color=colours_ml_trained, lw = 1)
plt.title("Area Under the ROC curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("AROC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(aroc_test) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(aroc_test)+0.1])
plt.ylim([0.75, 0.86])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
plt.close


# Plotting the auprc values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      auprc_train = []
      auprc_test = []
      for kfold in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
            auprc_train.append(np.load(f'{dir_in_temp}/auprc_train.npy'))
            auprc_test.append(np.load(f'{dir_in_temp}/auprc_test.npy'))
      plt.plot(np.arange(1,len(auprc_test) + 1), np.array(auprc_test), color=colours_ml_trained, lw = 2, label=ml_trained)
      plt.plot(np.arange(1,len(auprc_train) + 1), np.array(auprc_train), "--", color=colours_ml_trained, lw = 1)
plt.title("Area Under the Precision-Recall curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(auprc_test) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(auprc_test)+0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/auprc.png', dpi=1000)
plt.close


# Plotting the fb values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      fb_train = []
      fb_test = []
      for kfold in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
            fb_train.append(np.load(f'{dir_in_temp}/fb_train.npy'))
            fb_test.append(np.load(f'{dir_in_temp}/fb_test.npy'))
      plt.plot(np.arange(1,len(fb_test) + 1), np.array(fb_test), color=colours_ml_trained, lw = 2, label=ml_trained)
      plt.plot(np.arange(1,len(fb_train) + 1), np.array(fb_train), "--", color=colours_ml_trained, lw = 1)
plt.title("Area Under the Precision-Recall curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(fb_train) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(fb_train)+0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/fb.png', dpi=1000)
plt.close


# Plotting the best thresholds to convert the probabilistic events to yes-events
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      best_thr_train = []
      best_thr_test = []
      for kfold in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
            best_thr_train.append(np.load(f'{dir_in_temp}/best_thr_train.npy'))
            best_thr_test.append(np.load(f'{dir_in_temp}/best_thr_test.npy'))
      plt.plot(np.arange(1,len(best_thr_test) + 1), np.array(best_thr_test), color=colours_ml_trained, lw = 2, label=ml_trained)
      plt.plot(np.arange(1,len(best_thr_train) + 1), np.array(best_thr_train), "--", color=colours_ml_trained, lw = 1)
plt.title("Area Under the Precision-Recall curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(best_thr_test) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(best_thr_test)+0.1])
plt.ylim([0, 0.05])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/best_thr.png', dpi=1000)
plt.close


# Plotting the roc curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      plt.figure(figsize=(5, 5))
      for ind_k in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{ind_k + 1}'
            hr_train = np.load(f'{dir_in_temp}/hr_train.npy')
            hr_test = np.load(f'{dir_in_temp}/hr_test.npy')
            far_train = np.load(f'{dir_in_temp}/far_train.npy')
            far_test = np.load(f'{dir_in_temp}/far_test.npy')
            plt.plot(far_train, hr_train, "--", color=colours_ml_trained, lw = 1)
            plt.plot(far_test, hr_test, color=colours_ml_trained, lw = 2)
      plt.plot([0,1], [0,1], "--", color="#333333", lw = 1)
      plt.title("ROC curve", fontweight='bold', color="#333333", fontsize=14)
      plt.xlabel("False alarm rate", color = "#333333", fontsize = 12)
      plt.ylabel("Hit rate", color = "#333333", fontsize = 12)
      plt.tick_params(axis='x', colors='#333333', labelsize=12)
      plt.tick_params(axis='y', colors='#333333', labelsize=12)
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.grid(linewidth=0.5, color='gainsboro')
      plt.tight_layout()
      plt.savefig(f'{dir_out_temp}/roc_curve_{ml_trained}.png', dpi=1000)
      plt.close


# Plotting the reliability diagrams
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      plt.figure(figsize=(5, 5))
      for ind_k in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{ind_k + 1}'
            obs_freq_train = np.load(f'{dir_in_temp}/obs_freq_train.npy')
            obs_freq_test = np.load(f'{dir_in_temp}/obs_freq_test.npy')
            prob_pred_train = np.load(f'{dir_in_temp}/prob_pred_train.npy')
            prob_pred_test = np.load(f'{dir_in_temp}/prob_pred_test.npy')
            plt.plot(prob_pred_train, obs_freq_train, "--", color=colours_ml_trained, lw = 1)
            plt.plot(prob_pred_test, obs_freq_test, color=colours_ml_trained, lw = 2)
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
      plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}.png', dpi=1000)
      plt.close


