import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import shap

#########################################################################################################
# CODE DESCRIPTION
# 14_prob_ff_hydro_short_fc_shap.py computes the shap values for the short-range model predictions.

# Usage: python3 14_prob_ff_hydro_short_fc_shap.py

# Runtime: ~ 10 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# model_name (string): name of the model to train.
# loss_func_list (list of strings): type of loss function considered. Valid values are:
#                                                           - bce: no weights applied to loss function.
#                                                           - weighted_bce: wheight applied to loss function.
# eval_metric_list (list of strings): evaluation metric for the data-driven models. Valid values are:
#                                                           - auc: area under the roc curve.
#                                                           - auprc: area under the precion-recall curve.
# git_repo (string): repository's local path.
# dir_in_model (string): relative path of the directory containing the model to consider.
# file_in_pdt (string): relative path of the file containing the point data table to consider.
# dir_out (string): relative path of the directory containing the shap values.

##################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
model_name = "feed_forward_keras"
loss_func_list = ["bce", "weighted_bce"]
eval_metric_list = ["auc", "auprc"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_model = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
file_in_pdt_train = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
file_in_pdt_test = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
dir_out = "data/processed/14_prob_ff_hydro_short_fc_shap"
##################################################################################################


# Creating the verification plots
for loss_func in ["bce", "weighted_bce"]:

      for eval_metric in ["auc", "auprc"]:

            # Load the data-driven model
            print(f"Loding the data-driven model - {model_name} - for {loss_func} and {eval_metric}")
            file_in_model = f"{git_repo}/{dir_in_model}/{loss_func}/{eval_metric}/{model_name}" # future improvement: add the possibility to have the neural network
            if model_name == "feed_forward_keras":
                  model = load_model(f"{file_in_model}/model.h5", compile=False)
            else:
                  model = joblib.load(f"{file_in_model}/model.joblib")

            # Uploading the train/test dataset
            df = pd.read_csv(f"{git_repo}/{file_in_pdt_train}")
            X_train = df[feature_cols].copy()
            background_train = X_train.sample(1000, random_state=0).to_numpy()

            df = pd.read_csv(f"{git_repo}/{file_in_pdt_test}")
            X_test = df[feature_cols].copy()

            # Compute the SHAP values
            print("Computing the SHAP values")
            if model_name == "feed_forward_keras":
                  explainer = shap.Explainer(model,background_train)  
                  shap_values = explainer(X_test)
            else:
                  explainer = shap.TreeExplainer(model)
                  shap_values = explainer.shap_values(X_test)

            # Saving the SHAP values
            dir_out_temp = f"{git_repo}/{dir_out}/{loss_func}/{eval_metric}"
            file_out_temp = f"{dir_out_temp}/shap_{model_name}.pkl"
            os.makedirs(dir_out_temp, exist_ok=True)
            joblib.dump(shap_values, file_out_temp)