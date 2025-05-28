import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

##############################################################################################
# CODE DESCRIPTION
# 29_compute_shape.py computes the shap values for the model predictions.
# Runtime: the code takes up to 12 hours.

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# model_2_train_list (list of strings): names of the considered machine learning models. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# kfold_list (list of integers): list of kfolds containing the best set of hyperparameters for the correspondent model.
# git_repo (string): repository's local path.
# file_in_train (string): relative path of the file containing the training dataset.
# file_in_test (string): relative path of the file containing the test dataset.
# dir_out (string): relative path of the directory containing the final version of the trained machine learning models.

# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_50", "swvl", "sdfor", "lai"] 
target_col = "ff"
model_2_train_list = ["gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost", "random_forest_xgboost", "random_forest_lightgbm"]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_train = "data/compute/21_combine_pdt/pdt_2001_2020.csv"
dir_in_model = "data/compute/25_retrain_ml_test"
dir_out = "data/compute/29_shap"
##############################################################################################


# Upload the training dataset
df = pd.read_csv(git_repo + "/" + file_in_train)
X = df[feature_cols].copy()
y = df[target_col].copy()

# Compute the shap values
for model_2_train in model_2_train_list:

      # Define the output directory and file
      dir_out_temp = git_repo + "/" + dir_out + "/" + model_2_train
      file_out = dir_out_temp + "/shap.pkl"

      # Upload the considered model
      print("Upload the considered model")
      model = joblib.load(git_repo + "/" + dir_in_model + "/" + model_2_train + "/model.joblib")

      # Compute or read the SHAP values
      if not os.path.exists(file_out):
            print("Compute SHAP values")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            os.makedirs(dir_out_temp, exist_ok=True)
            joblib.dump(shap_values, file_out)
      else:
            print("Read SHAP values")
            shap_values = joblib.load(file_out)

      # # Create and show the SHAP summary plot
      # print("Create and show the SHAP summary plot")
      # shap.summary_plot(shap_values, X, plot_type="bar")
