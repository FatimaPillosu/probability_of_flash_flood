import os
import pandas as pd
import joblib
import pprint
import shap
import matplotlib.pyplot as plt

#########################################################################################################
# CODE DESCRIPTION
# 15_prob_ff_hydro_short_fc_shap.py computes the shap values for the short-range model predictions.

# Usage: python3 15_prob_ff_hydro_short_fc_shap.py

# Runtime: ~ 1 hour.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# git_repo (string): repository's local path.
# file_in_model (string): relative path of the file containing the model to consider.
# file_in_pdt (string): relative path of the file containing the point data table to consider.
# dir_out (string): relative path of the directory containing the shap values.

#########################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_50", "swvl", "sdfor", "lai"]
target_col = "ff"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_model = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold_new/gradient_boosting_xgboost/model.joblib"
file_in_pdt = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
dir_out = "data/processed/15_prob_ff_hydro_long_fc_shap"
#########################################################################################################


# Upload the considered model
print("Upload the considered model")
model = joblib.load(git_repo + "/" + file_in_model)

# Upload the point data table
print(f"\nUploading the point data table")
df = pd.read_csv(git_repo + "/" + file_in_pdt)
df = df.iloc[:1000]
X = df[feature_cols].copy()
y = df[target_col].copy()

# Define the output directory and file
dir_out_temp = git_repo + "/" + dir_out
file_out = dir_out_temp + "/shap.pkl"

# Compute or read the SHAP values
if not os.path.exists(file_out): # compute
      print("Compute SHAP values")
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(X)
      os.makedirs(dir_out_temp, exist_ok=True)
      joblib.dump(shap_values, file_out)
else: # read
      print("Read SHAP values")
      shap_values = joblib.load(file_out)

# Create and show the SHAP summary plot
print("Create and show the SHAP summary plot")
#shap.summary_plot(shap_values, X, plot_type="bar")
shap.dependence_plot("sdfor", shap_values, X)