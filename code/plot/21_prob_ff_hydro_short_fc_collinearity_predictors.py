import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#########################################################################
# CODE DESCRIPTION
# 21_prob_ff_hydro_short_fc_collinearity_predictors.py plots the collinearity existing 
# between the predictors considered in the data-driven model that computes the 
# short-range predictions for the areas at risk of flash floods.

# Usage: python3 21_prob_ff_hydro_short_fc_collinearity_predictors.py

# Runtime: ~ 1 minute.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# git_repo (string): repository's local path.
# file_in (string): relative path of the file containing the training dataset.
# dir_out (string): relative path of the directory containing the collinearity plot.

#########################################################################
# INPUT PARAMETERS
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
dir_out = "data/plot/21_prob_ff_hydro_short_fc_collinearity_predictors"
#########################################################################


# Reading the training dataset (point data table)
pdt = pd.read_csv(git_repo + "/" + file_in)
pred_list = ["tp_greater_0", "tp_prob_1", "tp_prob_2", "tp_prob_5", "tp_prob_10", "tp_prob_20", "tp_prob_50", "tp_prob_100", "tp_prob_max_1_adj_gb", "tp_prob_max_50_adj_gb", "swvl", "slor", "sdfor", "lai"]

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