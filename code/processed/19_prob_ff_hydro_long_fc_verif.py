import os
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

#########################################################################################################
# CODE DESCRIPTION
# 19_prob_ff_hydro_long_fc_verif.py computes the verification scores for the long-range forecasts of areas at risk of flash floods.

# Usage: python3 19_prob_ff_hydro_long_fc_verif.py

# Runtime: ~ 1 minute.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# step_f_start (integer, in hours): first final step of the accumulation period to consider. 
# step_f_final (integer, in hours): last final step of the accumulation period to consider. 
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# git_repo (string): repository's local path.
# file_in_model (string): relative path of the file containing the considered machine learning model.
# file_in_pdt (string): relative path of the files containing the point data tables for different lead times.
# dir_out (string): relative path of the directory containing the final version of the trained machine learning models.

#########################################################################################################
# INPUT PARAMETERS
step_f_start = 24
step_f_final = 120
feature_cols = ["tp_prob_1", "tp_prob_50", "swvl", "sdfor", "lai"]
target_col = "ff"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_model = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold_new/gradient_boosting_xgboost/model.joblib"
file_in_pdt = "data/processed/18_prob_ff_hydro_long_fc_combine_pdt/pdt_2021_2024"
dir_out = "data/processed/19_prob_ff_hydro_long_fc_verif"
#########################################################################################################


###################
# DATA LOADING UTIL #
###################

def load_data(csv_path: str, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
      
      """
      Loads a csv point data table (pdt) and returns X (features) and y (target).

      csv_path: 
            string representing the file path to the CSV data.
      feature_cols: 
            list of strings representing the pdt column names to be used as features.
      target_col: 
            string representing the pdt column name to be used as the target variable.
      return:
            tuple (X, y) where:
            - X is a pandas DataFrame containing the selected feature columns.
            - y is a pandas Series containing the target variable.

      Example:
      --------
      X, y = load_data(
            csv_path="/path/to/dataset.csv",
            feature_cols=["tp_prob_1", "tp_prob_20", "swvl", "sdfor", "lai"],
            target_col="ff"
      )

      Notes:
      - Any data cleaning or preprocessing steps should be done
            before loading the point data table in order to keep this code clean.
      - If the csv does not contain columns matching the names passed in
            feature_cols or target_col, a KeyError will be raised.
      """

      logger.info(f"Loading dataset from {csv_path}")
      df = pd.read_csv(csv_path)
      X = df[feature_cols].copy()
      y = df[target_col].copy()

      logger.info(f"Loaded dataset with shape X={X.shape}, y={y.shape}")

      return X, y

##############################################################################


# Load the data-driven model
model = joblib.load(f"{git_repo}/{file_in_model}")


# Computing the verification scores for the long-range forecasts
recall_all = []
f1_all = []
aroc_all = []
obs_freq_all = []
fc_pred_all = []
fb_all = []
far_all = []
hr_all = []
yes_event_thr_all = []

for step_f in range(step_f_start, step_f_final + 1, 24):

      # Read the point data table
      logger.info(f"\nReading the pdt for t+{step_f}")
      file_in_test = f"{git_repo}/{file_in_pdt}_{step_f:03d}.csv"
      X, y_obs = load_data(file_in_test, feature_cols, target_col)

      y_fc_prob = model.predict_proba(X)[:, 1]

      precision, recall_curve, thresholds = precision_recall_curve(y_obs, y_fc_prob)
      f1_curve = 2 * (precision * recall_curve) / (precision + recall_curve + 1e-6)
      best_idx = np.argmax(f1_curve)
      best_threshold = thresholds[best_idx]
      y_fc = (y_fc_prob >= best_threshold).astype(int)

      recall_all.append(recall_score(y_obs, y_fc))
      f1_all.append(f1_score(y_obs, y_fc))
      yes_event_thr_all.append(best_threshold)

      obs_freq, fc_pred = calibration_curve(y_obs, y_fc_prob, n_bins=100)
      obs_freq_all.append(obs_freq)
      fc_pred_all.append(fc_pred)

      ind_yes = np.where(y_fc == 1)[0]
      yes_fc = ind_yes.shape[0]
      yes_obs = np.sum(y_obs[ind_yes])
      fb_all.append( yes_fc / yes_obs )

      far, hr, _ = roc_curve(y_obs, y_fc_prob)
      far_all.append(far)
      hr_all.append(hr)
      aroc_all.append(auc(far, hr))
      
aroc_all = np.array(aroc_all)
f1_all = np.array(f1_all)
recall_all = np.array(recall_all)
yes_event_thr_all = np.array(yes_event_thr_all)
obs_freq_all = np.array(obs_freq_all, dtype=object)
fc_pred_all = np.array(fc_pred_all, dtype=object)
fb_all = np.array(fb_all)
far_all = np.array(far_all, dtype=object)
hr_all = np.array(hr_all, dtype=object)

# Saving the verification scores
dir_out_temp = f"{git_repo}/{dir_out}"
if not os.path.exists(dir_out_temp):
      os.makedirs(dir_out_temp)
np.save(dir_out_temp + "/obs_freq", obs_freq_all)
np.save(dir_out_temp + "/fc_pred", fc_pred_all)
np.save(dir_out_temp + "/fb", fb_all)
np.save(dir_out_temp + "/far", far_all)
np.save(dir_out_temp + "/hr", hr_all)
np.save(dir_out_temp + "/aroc", aroc_all)
np.save(dir_out_temp + "/recall", recall_all)
np.save(dir_out_temp + "/f1", f1_all)
np.save(dir_out_temp + "/yes_event_thr", yes_event_thr_all)