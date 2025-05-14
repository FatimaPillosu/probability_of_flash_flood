import os
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

################################################################################################################
# CODE DESCRIPTION
# 22_plot_collinearity_predictors.py plots the collinearity existing between the predictors considered in the data-driven flash flood model.
# Runtime: negligible.

# INPUT PARAMETERS DESCRIPTION
# git_repo (string): repository's local path.
# file_in (string): relative path of the file containing the training dataset.
# dir_out (string): relative path of the directory containing the collinearity plot.

# INPUT PARAMETERS
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in = "data/compute/21_combine_pdt/AllFF/pdt_AllFF_2005_2020.csv"
dir_out = "data/plot/22_collinearity_predictors"
###################################################################################


# Reading the training dataset (point data table)
pdt = pd.read_csv(git_repo + "/" + file_in)
pred_list = ["tp_prob_1", "tp_prob_2", "tp_prob_5", "tp_prob_10", "tp_prob_20", "tp_prob_50", "tp_prob_100", "swvl", "slor", "sdfor", "lai"]

# Computing the collinearity between the predictors, and displaying it as a heatmap
corr_matrix = pdt[pred_list].corr()
plt.figure(figsize=(9, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, cbar=True, square=True, fmt='.2f', annot_kws={"size": 7}, mask=mask, vmin=0, vmax=1)

# Saving the collinearity plot
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
      os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/collinearity.jpeg"
plt.savefig(file_out, format="jpeg", bbox_inches="tight", dpi=1000)
plt.close()

# Create scatter plots between different pairs of predictors
num_pred = len(pred_list)
fig, axes = plt.subplots(nrows=num_pred, ncols=num_pred, figsize=(20, 20))
for ax in axes.flatten(): # Remove x- and y-ticks
    ax.set_xticks([])
    ax.set_yticks([])
ind_pred_1 = 0
while ind_pred_1 < num_pred:
      pred_1 = pred_list[ind_pred_1]
      for ind_pred_2 in range(ind_pred_1 + 1, num_pred):
            pred_2 = pred_list[ind_pred_2]
            axes[ind_pred_2, ind_pred_1].scatter(pdt.loc[pdt["ff"] == 0, pred_1], pdt.loc[pdt["ff"] == 0, pred_2], color="silver", marker=".", alpha=0.4)
            axes[ind_pred_2, ind_pred_1].scatter(pdt.loc[pdt["ff"] == 1, pred_1], pdt.loc[pdt["ff"] == 1, pred_2], color="lightseagreen", marker=".", alpha=0.4)

# Save the scatter plots


      ind_pred_1 = ind_pred_1 + 1


# Saving the collinearity plot

file_out = dir_out_temp + "/collinearity.jpeg"
plt.savefig(file_out, format="jpeg", bbox_inches="tight", dpi=1000)
plt.close()





# y = pdt["ff"]
# X = pdt[pred_list] 
# sampler = SMOTE(k_neighbors=5, random_state=42)
# X_resampled, y_resampled = sampler.fit_resample(X, y)

# # Create scatter plots between different pairs of predictors
# num_pred = len(pred_list)
# fig, axes = plt.subplots(nrows=num_pred, ncols=num_pred, figsize=(20, 20))
# for ax in axes.flatten(): # Remove x- and y-ticks
#     ax.set_xticks([])
#     ax.set_yticks([])
# ind_pred_1 = 0
# while ind_pred_1 < num_pred:
#       pred_1 = pred_list[ind_pred_1]
#       for ind_pred_2 in range(ind_pred_1 + 1, num_pred):
#             pred_2 = pred_list[ind_pred_2]
#             axes[ind_pred_2, ind_pred_1].scatter(X_resampled.loc[y_resampled == 0, pred_1], X_resampled.loc[y_resampled == 0, pred_2], color="silver", marker=".", alpha=0.4)
#             axes[ind_pred_2, ind_pred_1].scatter(X_resampled.loc[y_resampled == 1, pred_1], X_resampled.loc[y_resampled == 1, pred_2], color="lightseagreen", marker=".", alpha=0.4)
#       ind_pred_1 = ind_pred_1 + 1
# plt.tight_layout()
# plt.show()