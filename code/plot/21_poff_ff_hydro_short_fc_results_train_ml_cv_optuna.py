import os
import numpy as np
import matplotlib.pyplot as plt

##############################################################################################################
# CODE DESCRIPTION
# 21_prob_ff_hydro_short_fc_results_train_ml_cv_optuna plots the verification results of the training of diverse machine learning models
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
# Note: It would have been more efficient to vectorise all the computations but, due to memory issues, it was not possible. Otherwise, 
# for the considered domain, we would have not been able to compute more than 100 bootstraps, a number that is well below the 
# recommended standars of at least 1000 repetitions. 

# Usage: python3 21_prob_ff_hydro_short_fc_results_train_ml_cv_optuna.py

# Runtime: negligible.

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

##############################################################################################################
# INPUT PARAMETERS
ml_trained_list = ["random_forest_xgboost", "random_forest_lightgbm", "gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost","feed_forward_keras"]
colours_ml_trained_list = ["coral", "limegreen", "mediumblue", "crimson", "gold", "dodgerblue"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna_old"
dir_out = "data/processed/21_prob_ff_hydro_short_fc_results_train_ml_cv_optuna"
##############################################################################################################

# Plotting the recall values
plt.figure(figsize=(6, 5))
for ind_ml, ml_trained in enumerate(ml_trained_list):

      colours_ml_trained = colours_ml_trained_list[ind_ml]
      
      dir_in_temp = f'{git_repo}/{dir_in}/{ml_trained}'
      recall = np.load(f'{dir_in_temp}/recall.npy')
      
      plt.plot(np.arange(len(recall)), recall, color=colours_ml_trained, lw = 3, label=ml_trained)
      plt.title("Recall", fontweight='bold', color="#333333", pad=60, fontsize=14)
      plt.xlabel("k_outer", color = "#333333", fontsize = 12)
      plt.ylabel("Recall", color = "#333333", fontsize = 12)
      plt.tick_params(axis='x', colors='#333333', labelsize=12)
      plt.tick_params(axis='y', colors='#333333', labelsize=12)
      plt.xticks(np.arange(len(recall)))
      plt.legend(loc='upper center', labelcolor='#333333', fontsize=10, bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False)
      plt.xlim([-0.1, len(recall)-0.9])
      plt.tight_layout()

plt.show()
exit()
