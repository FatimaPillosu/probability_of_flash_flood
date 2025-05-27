import os
import numpy as np
import matplotlib.pyplot as plt

##############################################################################################################
# CODE DESCRIPTION
# 22_prob_ff_hydro_short_fc_results_train_ml_cv_optuna plots the verification results of the training of diverse machine learning models
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
# Note: It would have been more efficient to vectorise all the computations but, due to memory issues, it was not possible. Otherwise, 
# for the considered domain, we would have not been able to compute more than 100 bootstraps, a number that is well below the 
# recommended standars of at least 1000 repetitions. 

# Usage: python3 22_prob_ff_hydro_short_fc_results_train_ml_cv_optuna.py

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
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

##############################################################################################################
# INPUT PARAMETERS
num_k_outer = 5
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "random_forest_lightgbm", "gradient_boosting_lightgbm", "gradient_boosting_catboost","feed_forward_keras"]
colours_ml_trained_list = ["mediumblue", "orangered", "limegreen", "crimson", "dodgerblue", "gold"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna_old"
dir_out = "data/plot/22_prob_ff_hydro_short_fc_results_train_ml_cv_optuna"
##############################################################################################################


# Creating the output directory
dir_out_temp = f'{git_repo}/{dir_out}'
os.makedirs(dir_out_temp, exist_ok=True)

# Plotting the recall values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      recall = np.load(f'{dir_in_temp}/recall.npy')
      plt.plot(np.arange(len(recall)), recall, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("Recall", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("Recall", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(recall)))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([-0.1, len(recall)-0.9])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/recall.png', dpi=1000)
plt.close


# Plotting the f1-score values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      f1 = np.load(f'{dir_in_temp}/f1.npy')
      plt.plot(np.arange(len(f1)), f1, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("F1-score", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("F1-score", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(f1)))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([-0.1, len(f1)-0.9])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/f1.png', dpi=1000)
plt.close


# Plotting the aroc values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      aroc = np.load(f'{dir_in_temp}/aroc.npy')
      plt.plot(np.arange(len(aroc)), aroc, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("Area Under the ROC curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("AROC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(aroc)))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([-0.1, len(aroc)-0.9])
plt.ylim([0.75, 0.9])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
plt.close


# Plotting the training times per fold
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      fold_time = np.load(f'{dir_in_temp}/fold_times.npy') / 60
      plt.plot(np.arange(len(fold_time)), fold_time, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("Training times", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("Times [hours]", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(len(fold_time)))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([-0.1, len(fold_time)-0.9])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/fold_times.png', dpi=1000)
plt.close


# Plotting the roc curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      
      plt.figure(figsize=(5, 5))
      for ind_k in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
            hr = np.load(f'{dir_in_temp}/hr_{ind_k+1}.npy')
            far = np.load(f'{dir_in_temp}/far_{ind_k+1}.npy')
            plt.plot(far, hr, color=colours_ml_trained, lw = 1)

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

      for ind_k in range(num_k_outer):
      
            dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
            prob_fc = np.load(f'{dir_in_temp}/fc_pred_{ind_k+1}.npy')
            freq_obs = np.load(f'{dir_in_temp}/obs_freq_{ind_k+1}.npy')
            
            plt.figure(figsize=(3, 3))
            plt.plot([0,1], [0,1], "--", color="#333333", lw = 1)
            plt.plot(prob_fc, freq_obs, color=colours_ml_trained, lw = 0.5)
            plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
            plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
            plt.tick_params(axis='x', colors='#333333', labelsize=10)
            plt.tick_params(axis='y', colors='#333333', labelsize=10)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.grid(linewidth=0.5, color='gainsboro')
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}_{ind_k + 1}.png', dpi=1000)
            plt.close()