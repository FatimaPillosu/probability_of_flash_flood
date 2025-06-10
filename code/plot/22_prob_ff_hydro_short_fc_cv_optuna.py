import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

######################################################################################
# CODE DESCRIPTION
# 22_prob_ff_hydro_short_fc_cv_optuna.py plots the results of Optuna's hyperparameter tuning.

# Usage: python3 22_prob_ff_hydro_short_fc_cv_optuna.py

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

######################################################################################
# INPUT PARAMETERS
num_k_outer = 5
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "gradient_boosting_catboost", "gradient_boosting_lightgbm", "random_forest_lightgbm", "gradient_boosting_adaboost", "feed_forward_keras"]
colours_ml_trained_list = ["mediumblue", "orangered", "teal", "crimson", "dodgerblue", "darkviolet", "magenta"]
eval_metric = "aroc"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/plot/22_prob_ff_hydro_short_fc_cv_optuna"
##############################################################################################################


# Plots about hyperparameters importance
meta_cols = {"number", "value", "state", "datetime_start", "datetime_complete", "wall_secs", "duration"}
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/optuna'
     
      for ind_k in range(num_k_outer):

            plt.figure(figsize=(7, 4))
            df = pd.read_csv(f'{dir_in_temp}/trials_rep1_fold{ind_k+1}.csv', delimiter = ",")
            hyperparam_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("datetime")]
            corr = {p: abs(np.corrcoef(df[p], df["value"])[0, 1]) for p in hyperparam_cols} # Pearson's r coefficient as proxy
            imp_temp = pd.Series(corr)
            imp = imp_temp / imp_temp.sum()
            imp = imp.sort_values()
            imp.index = imp.index.str.replace(r"^params_", "", regex=True)
            plt.barh(imp.index, [1] * len(imp_temp), color="whitesmoke", edgecolor="gainsboro", linewidth=0.5)
            plt.barh(imp.index, imp.values, color=colours_ml_trained)
            plt.title("Parameter importance", fontweight='bold', color="#333333", fontsize=14)
            plt.xlabel("Normalised abs(Pearson's r coefficient)", color = "#333333", fontsize = 12)
            plt.ylabel("Hyperparameters", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.tight_layout()

            dir_out_temp = f'{git_repo}/{dir_out}/{eval_metric}/{ml_trained}/fold_{ind_k + 1}'
            os.makedirs(dir_out_temp, exist_ok=True)        
            plt.savefig(f'{dir_out_temp}/param_importance{ind_k+1}.png', dpi=1000)
            plt.close()


# Plots about Optuna's optimisation history
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/optuna'
      
      for ind_k in range(num_k_outer):
           
            df = pd.read_csv(f'{dir_in_temp}/trials_rep1_fold{ind_k+1}.csv', delimiter = ",")
            
            plt.figure(figsize=(5, 4))
            plt.plot(df["number"]+1, df["value"], marker="o", markersize = 1, color=colours_ml_trained)
            plt.plot(df["number"]+1, df["value"].cummax(), linestyle="--", lw = 0.5, color=colours_ml_trained)
            plt.title("Optimisation history", fontweight='bold', color="#333333", fontsize=14)
            plt.xlabel("Trial number", color = "#333333", fontsize = 12)
            plt.ylabel(eval_metric, color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(df["number"] + 1)
            plt.xticks(df["number"]+1, [str(i) if (i % 5 == 0 or i == 1) else "" for i in df["number"]+1])  # label every 5th tick
            plt.tight_layout()

            dir_out_temp = f'{git_repo}/{dir_out}/{eval_metric}/{ml_trained}/fold_{ind_k + 1}'
            os.makedirs(dir_out_temp, exist_ok=True)     
            plt.savefig(f'{dir_out_temp}/opt_history_{ind_k+1}.png', dpi=1000)


# Plots Trial runtime vs. Model performance
for ind_ml, ml_trained in enumerate(ml_trained_list):
      
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      
      for ind_k in range(num_k_outer):
            
            dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/optuna'
            df = pd.read_csv(f'{dir_in_temp}/trials_rep1_fold{ind_k+1}.csv', delimiter = ",")
            
            plt.figure(figsize=(6, 5))
            trial_idx = df["number"] + 1
            sc = plt.scatter(df["wall_secs"], df["value"], c=trial_idx, cmap="plasma", norm=plt.Normalize(0, trial_idx.max()))

            plt.title("Trial runtime vs. Model performance", fontweight='bold', color="#333333", fontsize=14)
            plt.xlabel("Time [seconds]", color = "#333333", fontsize = 12)
            plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
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

            dir_out_temp = f'{git_repo}/{dir_out}/{eval_metric}/{ml_trained}/fold_{ind_k + 1}'
            os.makedirs(dir_out_temp, exist_ok=True)
            plt.savefig(f'{dir_out_temp}/runtime_performance_{ind_k+1}.png', dpi=1000)
            plt.close()
