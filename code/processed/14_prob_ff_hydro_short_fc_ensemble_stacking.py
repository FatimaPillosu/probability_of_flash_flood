import os
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
import xgboost
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

##############################################################################################################################################
# CODE DESCRIPTION
# 14_prob_ff_hydro_short_fc_ensemble_stacking.py creates the ensemble stacking for the chosen base models. 

# Usage: python3 14_prob_ff_hydro_short_fc_ensemble_stacking.py

# Runtime: the code takes up to 2 hours.

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# base_model_name_list (list of strings): names of the considered machine learning models. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# meta_model (string): name of the chosen meta-model.
# git_repo (string): repository's local path.
# file_in_train (string): relative path of the file containing the training dataset.
# file_in_test (string): relative path of the file containing the test dataset.
# dir_in (string): relative path of the directory containing the base models.
# dir_out (string): relative path of the directory containing the trained meta model.

##############################################################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_50", "swvl", "sdfor", "lai"] 
target_col = "ff"
base_model_name_list = ["gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost", "random_forest_xgboost", "random_forest_lightgbm"]
meta_model = "gradient_boosting_xgboost"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_train = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
file_in_test = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2023.csv"
dir_in = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
dir_out = "data/processed/14_prob_ff_hydro_short_fc_ensemble_stacking"
##############################################################################################################################################


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

print("XGBoost version:", xgboost.__version__)
print("LightGBM version:", lightgbm.__version__)
exit()



# Read the training and the test datasets
print(f"Reading the training dataset")
file_in_train = git_repo + "/" + file_in_train
X_train, y_train = load_data(file_in_train, feature_cols, target_col)

print(f"Reading the test dataset")
file_in_test = git_repo + "/" + file_in_test
X_test, y_test = load_data(file_in_test, feature_cols, target_col)

# Creating the base models for the ensemble stacking
print("Creating the base models for the ensemble stacking")
feature_meta_model_train = []
feature_meta_model_test = []
for base_model_name in base_model_name_list:
      file_base_model = git_repo + "/" + dir_in + "/" + base_model_name + "/model.joblib"
      base_model = joblib.load(file_base_model) 
      feature_meta_model_train.append(base_model.predict(X_train))
      feature_meta_model_test.append(base_model.predict(X_test))

feature_meta_model_train = pd.DataFrame(np.column_stack(feature_meta_model_train), columns=base_model_name_list)
feature_meta_model_test = pd.DataFrame(np.column_stack(feature_meta_model_test), columns=base_model_name_list)

# Training the meta-model
print("Training the meta-model")
meta_model = LogisticRegression()
meta_model.fit(feature_meta_model_test, y_test)
y_pred_prob = meta_model.predict_proba(feature_meta_model_train)[:, 1]

precision, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_curve = 2 * (precision * recall_curve) / (precision + recall_curve + 1e-6)
best_idx = np.argmax(f1_curve)
y_pred = (y_pred_prob >= thresholds[best_idx]).astype(int)

# Testing the model
print("Testing the model")
obs_freq, fc_pred = calibration_curve(y_test, y_pred_prob, n_bins=100)
far, hr, thr_roc = roc_curve(y_test, y_pred_prob)
aroc = auc(far, hr)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(recall)
print(f1)

plt.plot(fc_pred, obs_freq, lw = 0.5, marker=".", linestyle="-")
plt.plot([0, 1], [0, 1], linestyle="-", color="gray")
plt.xlabel("Forecast probability")
plt.ylabel("Observed frequency")
plt.title("Reliability Diagram - Ensemble")
plt.show()

plt.plot(far, hr, lw=0.5, label="AROC = %0.5f" % aroc)
plt.plot([-0.01, 1.01], [-0.01, 1.01], linestyle="-", color="gray")  
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Alarm Rate")
plt.ylabel("Hit Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()