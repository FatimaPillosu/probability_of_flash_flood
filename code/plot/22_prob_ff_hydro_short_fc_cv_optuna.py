import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################################################################################
# CODE DESCRIPTION
# 22_prob_ff_hydro_short_fc_cv_optuna.py plots the results of Optuna's hyperparameter tuning.

# Usage: python3 22_prob_ff_hydro_short_fc_cv_optuna.py

# Runtime: ~ 1 minute.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# num_rep (positive integer): number of repetitions considered in the nested cross-validation.
# num_k_outer (positive integer): number of outer k-folds considered in the nested cross-validation.
# ml_trained_list (list of strings): names of the models to train. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - feed_forward_keras
# colours_ml_trained_list (list of strings): list of colours to associate to each trained model.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

######################################################################################
# INPUT PARAMETERS
num_rep = 1
num_k_outer = 5
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "gradient_boosting_catboost", "gradient_boosting_lightgbm", "random_forest_lightgbm", "feed_forward_keras"]
colours_ml_trained_list = ["mediumblue", "teal", "crimson", "dodgerblue", "darkviolet", "magenta"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/plot/22_prob_ff_hydro_short_fc_cv_optuna"
##############################################################################################################


for loss_func in ["bce", "weighted_bce"]:

      for eval_metric in ["auc", "auprc"]:

            print(f"\nCreating Optuna plots for loss_fun = {loss_func}, and eval_metric = {eval_metric}")

            # Plots about model's generalisation capabilities under imbalanced datasets
            print(" - Plots about model's generalisation capabilities under imbalanced datasets")
            if eval_metric == "auc":
                  ylim = [0.5,1]
            elif eval_metric == "auprc":
                  ylim = [0,0.04]
            
            for ind_ml, ml_trained in enumerate(ml_trained_list):
                  
                  colours_ml_trained = colours_ml_trained_list[ind_ml]
                  dir_in_temp = f'{git_repo}/{dir_in}/{loss_func}/{eval_metric}/{ml_trained}/optuna'

                  plt.figure(figsize=(4, 4))

                  for ind_rep in range(num_rep):
                        
                        mean = []
                        median = []
                        max = []
                        min = []
                        
                        for ind_k in range(num_k_outer):
                              
                              df = pd.read_csv(f'{dir_in_temp}/trials_rep{ind_rep + 1}_fold{ind_k + 1}.csv', delimiter = ",")
                              eval_metric_vals = df["value"]
                              mean.append(eval_metric_vals.mean())
                              median.append(eval_metric_vals.median())
                              max.append(eval_metric_vals.max())
                              min.append(eval_metric_vals.min())
                              
                        plt.plot(np.arange(1, num_k_outer + 1), np.array(mean), color = colours_ml_trained, lw = 2)
                        plt.plot(np.arange(1, num_k_outer + 1), np.array(median), "--", color = colours_ml_trained, lw = 2)
                        plt.fill_between(np.arange(1, num_k_outer + 1), np.array(min), np.array(max, ), color = colours_ml_trained, alpha=0.4, edgecolor="none")
                        plt.xlabel("Outer folds", color = "#333333", fontsize = 12)
                        plt.ylabel(eval_metric.upper(), color = "#333333", fontsize = 12)
                        plt.tick_params(axis='x', colors='#333333', labelsize=12)
                        plt.tick_params(axis='y', colors='#333333', labelsize=12)
                        plt.ylim(ylim)
                        plt.tight_layout()

                        dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}/{ml_trained}'
                        os.makedirs(dir_out_temp, exist_ok=True)        
                        plt.savefig(f'{dir_out_temp}/model_generalisation.png', dpi=1000)
                        plt.close()


            # Plots about average model's training times
            print(" - Plots about average model's training times")
            for ind_ml, ml_trained in enumerate(ml_trained_list):
                  
                  colours_ml_trained = colours_ml_trained_list[ind_ml]
                  dir_in_temp = f'{git_repo}/{dir_in}/{loss_func}/{eval_metric}/{ml_trained}/optuna'

                  plt.figure(figsize=(4, 4))
                  
                  for ind_rep in range(num_rep):

                        mean = []
                        max = []
                        min = []

                        for ind_k in range(num_k_outer):
                              
                              df = pd.read_csv(f'{dir_in_temp}/trials_rep{ind_rep + 1}_fold{ind_k+1}.csv', delimiter = ",")
                              time_vals = df["wall_secs"]
                              mean.append(time_vals.mean())
                              max.append(time_vals.max())
                              min.append(time_vals.min())
                              
                        plt.plot(np.arange(1, num_k_outer + 1), np.array(mean), color = colours_ml_trained, lw = 2)
                        plt.fill_between(np.arange(1, num_k_outer + 1), np.array(min), np.array(max, ), color = colours_ml_trained, alpha=0.4, edgecolor="none")
                        plt.xlabel("Outer folds", color = "#333333", fontsize = 12)
                        plt.ylabel("Times [seconds]", color = "#333333", fontsize = 12)
                        plt.tick_params(axis='x', colors='#333333', labelsize=12)
                        plt.tick_params(axis='y', colors='#333333', labelsize=12)
                        plt.ylim([0,2000])
                        plt.tight_layout()
                        
                        dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}/{ml_trained}'
                        os.makedirs(dir_out_temp, exist_ok=True)        
                        plt.savefig(f'{dir_out_temp}/training_time.png', dpi=1000)
                        plt.close()


            # Plots about hyperparameters importance
            print(" - Plots about hyperparameters importance")
            meta_cols = {"number", "value", "state", "datetime_start", "datetime_complete", "duration", "wall_secs", }
            for ind_ml, ml_trained in enumerate(ml_trained_list):
                  
                  colours_ml_trained = colours_ml_trained_list[ind_ml]
                  dir_in_temp = f'{git_repo}/{dir_in}/{loss_func}/{eval_metric}/{ml_trained}/optuna'
            
                  imp = []
                  plt.figure(figsize=(5, 4))

                  for ind_rep in range(num_rep):

                        for ind_k in range(num_k_outer):

                              df = pd.read_csv(f'{dir_in_temp}/trials_rep{ind_rep + 1}_fold{ind_k+1}.csv', delimiter = ",")
                              hyperparam_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("datetime")]
                              corr = {p: abs(np.corrcoef(df[p], df["value"])[0, 1]) for p in hyperparam_cols}
                              imp_temp = pd.Series(corr)
                              imp.append(imp_temp / imp_temp.sum())
                  
                        imp_df = pd.concat(imp, axis=1)          
                        mean_imp = imp_df.mean(axis=1)
                        mean_imp    = mean_imp.sort_values(ascending=True)
                        
                        max_imp = imp_df.max(axis=1)[mean_imp.index] - mean_imp
                        min_imp = mean_imp - imp_df.min(axis=1)[mean_imp.index]
                        err_imp = np.vstack([min_imp, max_imp])

                        mean_imp.index = mean_imp.index.str.replace(r"^params_", "", regex=True)
                        plt.barh(mean_imp.index, [1] * len(mean_imp.values), color="whitesmoke", edgecolor="gainsboro", linewidth=0.5)
                        plt.barh(mean_imp.index, mean_imp.values, xerr=err_imp, color=colours_ml_trained, error_kw={
                        "ecolor":   "#888888",
                        "elinewidth": 1.2,
                        "capsize": 4,
                        "capthick": 1.2
                  })
                        plt.xlabel("Normalised abs(Pearson's r coefficient)", color = "#333333", fontsize = 12)
                        plt.tick_params(axis='x', colors='#333333', labelsize=12)
                        plt.tick_params(axis='y', colors='#333333', labelsize=12)
                        plt.tight_layout()

                        dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}/{ml_trained}'
                        os.makedirs(dir_out_temp, exist_ok=True)        
                        plt.savefig(f'{dir_out_temp}/param_importance.png', dpi=1000)
                        plt.close()


            # Plots about Optuna's optimisation history
            print(" - Plots about Optuna's optimisation history")
            for ind_ml, ml_trained in enumerate(ml_trained_list):
                  
                  colours_ml_trained = colours_ml_trained_list[ind_ml]
                  dir_in_temp = f'{git_repo}/{dir_in}/{loss_func}/{eval_metric}/{ml_trained}/optuna'
                  
                  plt.figure(figsize=(4, 4))

                  for ind_rep in range(num_rep):

                        for ind_rep in range(num_rep):

                              alpha = [0.1, 0.2, 0.4, 0.6, 0.9]
                              for ind_k in range(num_k_outer):
                              
                                    df = pd.read_csv(f'{dir_in_temp}/trials_rep{ind_rep + 1}_fold{ind_k+1}.csv', delimiter = ",")
                                    
                                    plt.plot(df["number"]+1, df["value"], "o--", markersize=2, lw=1, color="#333333", alpha = alpha[ind_k])
                                    plt.plot(df["number"]+1, df["value"].cummax(), linestyle="-", lw = 3, color=colours_ml_trained, alpha = alpha[ind_k])
                              
                              plt.xlabel("Trial number", color = "#333333", fontsize = 12)
                              plt.ylabel(eval_metric, color = "#333333", fontsize = 12)
                              plt.tick_params(axis='x', colors='#333333', labelsize=12)
                              plt.tick_params(axis='y', colors='#333333', labelsize=12)
                              plt.ylim([0,0.04])
                              plt.xticks(df["number"] + 1)
                              plt.xticks(df["number"]+1, [str(i) if (i % 5 == 0 or i == 1) else "" for i in df["number"]+1])  # label every 5th tick
                              plt.tight_layout()

                              dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}/{ml_trained}'
                              os.makedirs(dir_out_temp, exist_ok=True)     
                              plt.savefig(f'{dir_out_temp}/optuna_history.png', dpi=1000)
                              plt.close()