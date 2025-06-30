import os
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


##############################################################################################################
# CODE DESCRIPTION
# 23_prob_ff_hydro_short_fc_optuna_verify.py plots the verification results of the training of diverse machine learning models
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall relaibility)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
#     - area under the precision-recall curve (overall performance)
#     - precision-recall curve (breakdown score for imbalanced datasets)

# Usage: python3 23_prob_ff_hydro_short_fc_optuna_verify.py

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
# num_rep (positive integer): number of repetitions in the nested cross-validation.
# num_kfold_outer (positive integer): number of outer k-folds considered in the nested cross-validation.
# loss_fn_choice (string): name of the loss function to consider in the plots. Valid values are:
#                                               - bce: binary crossentropy
#                                               - weighted_bce: weighted binary crossentropy (specific for imbalanced datasets)
# eval_metric (string): evaluation metric used for hyperparameter tuning by Optuna. Valid values are:
#                                         - auc: area under the ROC curve.
#                                         - auprc: area under the precision-recall curve (specific for imbalanced datasets)
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

##############################################################################################################
# INPUT PARAMETERS
ml_trained_list = ["gradient_boosting_xgboost", "random_forest_xgboost", "gradient_boosting_catboost", "gradient_boosting_lightgbm", "random_forest_lightgbm", "feed_forward_keras"]
colours_ml_trained_list = ["mediumblue", "coral", "lightseagreen", "deeppink", "dodgerblue", "mediumorchid", "magenta"]
num_rep = 1
num_kfold_outer = 5
loss_fn_choice = "bce"
eval_metric = "auc"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/plot/23_prob_ff_hydro_short_fc_optuna_verify"
##############################################################################################################


# Creating the input/output directory
dir_in_temp = f'{git_repo}/{dir_in}/{loss_fn_choice}/{eval_metric}'
dir_out_temp = f'{git_repo}/{dir_out}/{loss_fn_choice}/{eval_metric}'
os.makedirs(dir_out_temp, exist_ok=True)

rep_list = np.arange(1, num_rep + 1)
kfold_outer_list = np.arange(1, num_kfold_outer + 1)

# Define which models perform better
plt.figure(figsize=(6, 5))
metric = []
for ind_ml, ml_trained in enumerate(ml_trained_list):
      for rep in rep_list:
            aroc = []
            for kfold_outer in kfold_outer_list:
                  aroc.append(np.load(f'{dir_in_temp}/{ml_trained}/aroc_rep{rep}_fold{kfold_outer}.npy'))
      metric.append(aroc)

data = pd.DataFrame({'gradient_boosting_xgboost': metric[0], 'random_forest_xgboost': metric[1], 'gradient_boosting_catboost': metric[2], 'gradient_boosting_lightgbm': metric[3], 'random_forest_lightgbm': metric[4], 'feed_forward_keras': metric[5]})
pvals = sp.posthoc_nemenyi_friedman(data)
binary_mask = (pvals >= 0.05).astype(int)
cmap = ListedColormap(['blue', 'red'])
sns.heatmap(binary_mask, annot=pvals.round(3), cmap=cmap, cbar=False, linewidths=0.5, linecolor='black')
plt.show()
exit()




# Plotting the aroc values
plt.figure(figsize=(6, 5))
av_aroc = []
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      for rep in rep_list:
            aroc = []
            for kfold_outer in kfold_outer_list:
                  aroc.append(np.load(f'{dir_in_temp}/{ml_trained}/aroc_rep{rep}_fold{kfold_outer}.npy'))
      # plt.plot(kfold_outer_list, np.array(aroc), color = colours_ml_trained, lw = 2, label=ml_trained)
      av_aroc.append(np.mean(aroc))

import seaborn as sns
import pandas as pd

# Prepare DataFrame for Seaborn
df = pd.DataFrame([aroc])

sns.violinplot(data=df, x="Model", y="AUC", inner="box")
plt.title('Model AUC Distributions Across Folds')
plt.show()
exit()







sorted_data = sorted(zip(av_aroc, ml_trained_list, colours_ml_trained_list), reverse=False)
values_sorted, categories_sorted, colors_sorted = zip(*sorted_data)
plt.barh(categories_sorted, values_sorted, color=colors_sorted)
# plt.xlim([0.79, 0.9])
plt.yticks([])
plt.show()
exit()







plt.ylabel("AROC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12, )
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
plt.close


# Plotting the auprc values
plt.figure(figsize=(6, 5))
av_auprc = []
for ind_ml, ml_trained in enumerate(ml_trained_list):
      colours_ml_trained = colours_ml_trained_list[ind_ml]
      for rep in rep_list:
            auprc = []
            for kfold_outer in kfold_outer_list:
                  auprc.append(np.load(f'{dir_in_temp}/{ml_trained}/auprc_rep{rep}_fold{kfold_outer}.npy'))
      av_auprc.append(np.mean(auprc))
      plt.plot(kfold_outer_list, np.array(auprc), color = colours_ml_trained, lw = 3, label=ml_trained)

# sorted_data = sorted(zip(av_auprc, ml_trained_list, colours_ml_trained_list), reverse=False)
# values_sorted, categories_sorted, colors_sorted = zip(*sorted_data)
# plt.barh(categories_sorted, values_sorted, color=colors_sorted)
# plt.yticks([])
# plt.show()
# exit()


plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
plt.tick_params(axis='x', colors='#333333', labelsize=12, )
plt.tick_params(axis='y', colors='#333333', labelsize=12)
plt.grid(axis='y', linewidth=0.5, color='gainsboro')
plt.show()
exit()
plt.savefig(f'{dir_out_temp}/auprc.png', dpi=1000)
plt.close


# # Plotting the fb values
# plt.figure(figsize=(6, 5))
# score_train = []
# score = []
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       dir_in_temp = f'{git_repo}/{dir_in}/{eval_metric}/{ml_trained}/fold{kfold + 1}'
#       score_train.append(np.load(f'{dir_in_temp}/fb_train.npy'))
#       score.append(np.load(f'{dir_in_temp}/fb_test.npy'))
# plt.plot(np.array(ml_trained_list), np.array(score), color = "grey", lw = 2, label=ml_trained)
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
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       best_thr = np.load(f'{dir_in_temp}/{ml_trained}/best_threshold.npy')[0]
#       plt.plot(np.arange(len(best_thr)), best_thr, color = colours_ml_trained, lw = 2, label=ml_trained)
# plt.ylabel("Best Probability Threshold", color = "#333333", fontsize = 12)
# plt.tick_params(axis='x', colors='#333333', labelsize=12, )
# plt.tick_params(axis='y', colors='#333333', labelsize=12)
# plt.grid(axis='y', linewidth=0.5, color='gainsboro')
# plt.show()
# exit()
# plt.savefig(f'{dir_out_temp}/best_thr.png', dpi=1000)
# plt.close


# # Plotting the precision-recall curves
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       plt.figure(figsize=(5, 5))
#       dir_in_temp = f'{git_repo}/{dir_in}/{loss_fn_choice}/{eval_metric}/{ml_trained}'
#       precision_train = np.load(f'{dir_in_temp}/precision_rep1_fold1.npy')
#       #precision_test = np.load(f'{dir_in_temp}/precision_test.npy')
#       recall_train = np.load(f'{dir_in_temp}/recall_rep1_fold1.npy')
#       #recall_test = np.load(f'{dir_in_temp}/recall_test.npy')
#       plt.plot(recall_train, precision_train, "--", color=colours_ml_trained, lw = 0.5)
#       #plt.plot(recall_test, precision_test, color=colours_ml_trained, lw = 1)
#       plt.plot([0,1], [1,0], color="#333333", lw = 1)
#       plt.title("Precision-recall curve", fontweight='bold', color="#333333", fontsize=14)
#       plt.xlabel("Recall (also Hit Rate)", color = "#333333", fontsize = 12)
#       plt.ylabel("Precision", color = "#333333", fontsize = 12)
#       plt.tick_params(axis='x', colors='#333333', labelsize=12)
#       plt.tick_params(axis='y', colors='#333333', labelsize=12)
#       plt.xlim([-0.05, 1.05])
#       plt.ylim([-0.05, 1.05])
#       plt.grid(linewidth=0.5, color='gainsboro')
#       plt.tight_layout()
#       plt.show()
#       exit()
#       plt.savefig(f'{dir_out_temp}/precision_recall_curve_{ml_trained}.png', dpi=1000)
#       plt.close


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


# # Plotting the reliability diagrams
# for ind_ml, ml_trained in enumerate(ml_trained_list):
#       colours_ml_trained = colours_ml_trained_list[ind_ml]
#       plt.figure(figsize=(5, 5))
#       for kfold in range(num_kfold):
#             obs_freq = np.load(f'{dir_in_temp}/{ml_trained}/obs_freq_rep1_fold{kfold + 1}.npy')
#             prob_pred = np.load(f'{dir_in_temp}/{ml_trained}/prob_pred_rep1_fold{kfold + 1}.npy')
#             plt.plot(prob_pred, obs_freq, color=colours_ml_trained, lw = 1)
#             plt.plot([0,1], [0,1], "--", color="#333333", lw = 1)
#             plt.title("Reliability diagram", fontweight='bold', color="#333333", fontsize=14)
#             plt.xlabel("Forecast probability", color = "#333333", fontsize = 12)
#             plt.ylabel("Observation frequency", color = "#333333", fontsize = 12)
#             plt.tick_params(axis='x', colors='#333333', labelsize=12)
#             plt.tick_params(axis='y', colors='#333333', labelsize=12)
#             plt.xlim([-0.05, 1.05])
#             plt.ylim([-0.05, 1.05])
#             plt.grid(linewidth=0.5, color='gainsboro')
#             plt.tight_layout()
#             plt.show()
#             exit()

#       plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}.png', dpi=1000)
#       plt.close