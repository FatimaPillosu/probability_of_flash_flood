import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

##############################################################################################################
# CODE DESCRIPTION
# 22_prob_ff_hydro_short_fc_results_train_ml_cv_optuna.py plots the verification results of the training of diverse machine learning models
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
ml_trained_list = ["gradient_boosting_xgboost"]
colours_ml_trained_list = ["mediumblue", "orangered", "teal", "crimson", "dodgerblue", "darkviolet"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/plot/22_prob_ff_hydro_short_fc_results_train_ml_cv_optuna"
##############################################################################################################


# Creating the output directory
dir_out_temp = f'{git_repo}/{dir_out}'
os.makedirs(dir_out_temp, exist_ok=True)


# # Plots about Optuna's optimisation history
# for ind_ml, ml_trained in enumerate(ml_trained_list):
      
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}/optuna'

#       for ind_k in range(num_k_outer):
            
#             df = pd.read_csv(f'{dir_in_temp}/trials_rep1_fold{ind_k+1}.csv', delimiter = ",")

#             plt.figure(figsize=(5, 4))
#             plt.plot(df["number"]+1, df["value"], marker="o", markersize = 1, color=colours_ml_trained)
#             plt.plot(df["number"]+1, df["value"].cummax(), linestyle="--", lw = 0.5, color=colours_ml_trained)
#             plt.title("Optimisation history", fontweight='bold', color="#333333", fontsize=14)
#             plt.xlabel("Trial number", color = "#333333", fontsize = 12)
#             plt.ylabel("AROC", color = "#333333", fontsize = 12)
#             plt.tick_params(axis='x', colors='#333333', labelsize=12)
#             plt.tick_params(axis='y', colors='#333333', labelsize=12)
#             plt.ylim([0.75,0.9])
#             plt.xticks(df["number"] + 1)
#             plt.xticks(df["number"]+1, [str(i) if (i % 5 == 0 or i == 1) else "" for i in df["number"]+1])  # label every 5th tick
#             plt.yticks(np.arange(0.75, 0.9+0.01, 0.05))
#             plt.tight_layout()
#             plt.savefig(f'{dir_out_temp}/opt_history_{ind_k+1}.png', dpi=1000)


# # Plots about parameter importance
# for ind_ml, ml_trained in enumerate(ml_trained_list):
      
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}/optuna'
#       meta_cols = {"number", "value", "state", "datetime_start", "datetime_complete", "wall_secs", "duration"}

#       for ind_k in range(num_k_outer):
            
#             df = pd.read_csv(f'{dir_in_temp}/trials_rep1_fold{ind_k+1}.csv', delimiter = ",")
#             hyperparam_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("datetime")]
#             corr = {p: abs(np.corrcoef(df[p], df["value"])[0, 1]) for p in hyperparam_cols} # Pearson's r coefficient as proxy
#             imp_temp = pd.Series(corr)
#             imp = imp_temp / imp_temp.sum()
#             imp = imp.sort_values()
#             imp.index = imp.index.str.replace(r"^params_", "", regex=True)
#             plt.figure(figsize=(7, 4))
#             plt.barh(imp.index, [1] * len(imp_temp), color="whitesmoke", edgecolor="gainsboro", linewidth=0.5)
#             plt.barh(imp.index, imp.values, color=colours_ml_trained)
#             plt.title("Parameter importance", fontweight='bold', color="#333333", fontsize=14)
#             plt.xlabel("Normalised abs(Pearson's r coefficient)", color = "#333333", fontsize = 12)
#             plt.ylabel("Hyperparameters", color = "#333333", fontsize = 12)
#             plt.tick_params(axis='x', colors='#333333', labelsize=12)
#             plt.tick_params(axis='y', colors='#333333', labelsize=12)
#             plt.tight_layout()
#             plt.savefig(f'{dir_out_temp}/param_imp{ind_k+1}.png', dpi=1000)


# Plots about parameter importance
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}/optuna'
      for ind_k in range(num_k_outer):
            df = pd.read_csv(f'{dir_in_temp}/trials_rep1_fold{ind_k+1}.csv', delimiter = ",")
            trial_idx = df["number"] + 1
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(df["wall_secs"], df["value"], c=trial_idx, cmap="plasma", norm=plt.Normalize(0, trial_idx.max()))
            plt.title("Trial runtime vs. Model performance", fontweight='bold', color="#333333", fontsize=14)
            plt.xlabel("Time [seconds]", color = "#333333", fontsize = 12)
            plt.ylabel("AROC", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.ylim([0.75,0.9])
            plt.yticks(np.arange(0.75, 0.9+0.01, 0.05))
            cbar = plt.colorbar(sc)
            cbar.set_label("Trial number", fontsize=12, color="#333333")
            step  = 2
            ticks = np.arange(0, trial_idx.max() + 1, step)
            if ticks[-1] != trial_idx.max():    # ensure the very last tick is the max
                  ticks = np.append(ticks, trial_idx.max())
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(colors="#333333", labelsize=11)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/runtime_performance_{ind_k+1}.png', dpi=1000)


# Plotting the recall values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      recall = np.load(f'{dir_in_temp}/recall.npy')[0]
      plt.plot(np.arange(1,len(recall) + 1), recall, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("Recall", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("Recall", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(recall) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(recall)+0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/recall.png', dpi=1000)
plt.close


# Plotting the f1-score values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      f1 = np.load(f'{dir_in_temp}/f1.npy')[0]
      plt.plot(np.arange(1,len(f1) + 1), f1, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("F1-score", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("F1-score", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(f1) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(f1)+0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/f1.png', dpi=1000)
plt.close


# Plotting the aroc values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      aroc = np.load(f'{dir_in_temp}/aroc.npy')[0]
      plt.plot(np.arange(1,len(aroc) + 1), aroc, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("Area Under the ROC curve", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("AROC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(aroc) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(aroc)+0.1])
plt.ylim([0.75, 0.9])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
plt.close


# Plotting the best thresholds to define yes-events
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      best_threshold = np.load(f'{dir_in_temp}/best_threshold.npy')[0]
      plt.plot(np.arange(1,len(best_threshold) + 1), best_threshold, color=colours_ml_trained, lw = 2, label=ml_trained)
plt.title("Best thresholds to define yes-events", fontweight='bold', color="#333333", pad=60, fontsize=14)
plt.xlabel("k_outer", color = "#333333", fontsize = 12)
plt.ylabel("Probability [%]", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12)
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.xticks(np.arange(1,len(best_threshold) + 1))
plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
plt.xlim([0.9, len(best_threshold)+0.1])
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.tight_layout()
plt.savefig(f'{dir_out_temp}/best_thresholds.png', dpi=1000)
plt.close


# Plotting the roc curves
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      
      plt.figure(figsize=(5, 5))
      for ind_k in range(num_k_outer):
            dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
            hr = np.load(f'{dir_in_temp}/hr_rep1_fold{ind_k+1}.npy')
            far = np.load(f'{dir_in_temp}/far_rep1_fold{ind_k+1}.npy')
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
            prob_fc = np.load(f'{dir_in_temp}/prob_pred_rep1_fold{ind_k+1}.npy')
            freq_obs = np.load(f'{dir_in_temp}/obs_freq_rep1_fold{ind_k+1}.npy')
            
            plt.figure(figsize=(3, 3))
            plt.plot([0,1], [0,1], "--", color="#333333", lw = 1)
            plt.plot(prob_fc, freq_obs, color=colours_ml_trained, lw = 0.5)
            plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
            plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
            plt.tick_params(axis='x', colors='#333333', labelsize=10)
            plt.tick_params(axis='y', colors ='#333333', labelsize=10)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.grid(linewidth=0.5, color='gainsboro')
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}_{ind_k + 1}.png', dpi=1000)
            plt.close()