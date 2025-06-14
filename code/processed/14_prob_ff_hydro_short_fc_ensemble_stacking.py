import os
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
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
# eval_metric (string): evaluation metric used for hyperparameter tuning.
# base_model_name_list (list of strings): names of the considered machine learning models. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# kfold_count (integer): count of outter k-folds considering during the hyperparameters tunning.
# meta_model (string): name of the chosen meta-model.
# git_repo (string): repository's local path.
# file_in_train (string): relative path of the file containing the training dataset.
# file_in_test (string): relative path of the file containing the test dataset.
# dir_in (string): relative path of the directory containing the base models.
# dir_out (string): relative path of the directory containing the trained meta model.

##############################################################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
eval_metric = "auprc"
base_model_name_list = ["random_forest_xgboost", "random_forest_lightgbm", "gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost"]
kfold = 1
meta_model = "gradient_boosting_xgboost"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_train = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
file_in_test = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
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


##################
# VERIFICATION UTIL #
##################

def verification(y_test, pred_prob_test):

      precision, recall, thresholds = precision_recall_curve(y_test, pred_prob_test)
      f1_curve = 2 * (precision * recall) / (precision + recall + 1e-6)
      best_idx = np.argmax(f1_curve)
      pred_test = (pred_prob_test >= thresholds[best_idx]).astype(int)

      obs_freq, fc_pred = calibration_curve(y_test, pred_prob_test, n_bins=100)
      far, hr, _ = roc_curve(y_test, pred_prob_test)
      auprc = average_precision_score(y_test, pred_prob_test)
      aroc = auc(far, hr)
      fb = pred_test.sum() / max(y_test[pred_test == 1].sum(), 1)

      return precision, recall, obs_freq, fc_pred, far, hr, auprc, aroc, fb


##############################################################################


# Create the output directory
dir_out_temp = git_repo + "/" + dir_out + "/" + eval_metric
os.makedirs(dir_out_temp, exist_ok=True)


# Read the training and the test datasets
print(f"\nReading the training dataset")
file_in_train = git_repo + "/" + file_in_train
X_train, y_train = load_data(file_in_train, feature_cols, target_col)

print(f"\nReading the test dataset")
file_in_test = git_repo + "/" + file_in_test
X_test, y_test = load_data(file_in_test, feature_cols, target_col)


# Creating or reading the base models for the ensemble stacking
file_base_models_train = f"{dir_out_temp}/base_models_train.csv"
file_base_models_test = f"{dir_out_temp}/base_models_test.csv"

feature_meta_model_train = []
feature_meta_model_test = []
col_names_list = []
for base_model_name in base_model_name_list:
      print(f"\nCreating the base models from: " + base_model_name + " - fold n." + str(kfold + 1))
      file_base_model = git_repo + "/" + dir_in + "/" + eval_metric + "/" + base_model_name + "/fold" + str(kfold) + "/model.joblib"
      base_model = joblib.load(file_base_model)
      feature_meta_model_train.append(base_model.predict_proba(X_train)[:, 1])
      feature_meta_model_test.append(base_model.predict_proba(X_test)[:, 1])
      col_names_list.append(base_model_name + "_fold" + str(kfold))
feature_meta_model_train = pd.DataFrame(np.column_stack(feature_meta_model_train), columns=col_names_list)
feature_meta_model_test = pd.DataFrame(np.column_stack(feature_meta_model_test), columns=col_names_list)

# Training the meta-model (XGBoost) with hyperparameter optimisation with Optuna
print(f"\nTraining the meta-model")

# X_tr, X_val, y_tr, y_val = train_test_split(
#     feature_meta_model_train, y_train, test_size=0.2,
#     stratify=y_train, random_state=42
# )

# def objective(trial):
      
#       logger.info(f"★ Trial {trial.number} started")

#       params = {
#             "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
#             "max_depth":        trial.suggest_int("max_depth", 2, 8),
#             "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#             "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#             "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
#             "min_child_weight": trial.suggest_float("min_child_weight", 0.0, 10.0),
#             "objective":        "binary:logistic",
#             "eval_metric":      "aucpr",
#             "random_state":     42,
#             "n_jobs":           4,
#             }
      
#       model = XGBClassifier(**params)
#       model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
#       y_val_prob = model.predict_proba(X_val)[:, 1]
      
#       return average_precision_score(y_val, y_val_prob)

# study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(objective, n_trials=5)

# best_params = study.best_params
# best_params.update({
#     "objective": "binary:logistic",
#     "eval_metric": "aucpr",
#     "random_state": 42,
#     "n_jobs": 4
# })

meta_model = XGBClassifier()
meta_model.fit(feature_meta_model_train, y_train)
pred_prob_test_es = meta_model.predict_proba(feature_meta_model_test)[:, 1]

# joblib.dump(meta_model, os.path.join(dir_out_temp, "meta_model.joblib"))
# with open(os.path.join(dir_out_temp, "best_params.json"), "w") as fp:
#       json.dump(best_params, fp, indent=2)


# Testing the model
print(f"\nTesting the blended multi-model")
precision_es, recall_es, obs_freq_es, fc_pred_es, far_es, hr_es, auprc_es, aroc_es, fb_es = verification(y_test, pred_prob_test_es)

print(f"AROC_ES = {aroc_es}")
print(f"AUPRC_ES = {auprc_es}")
print(f"FB_ES = {fb_es}")

plt.plot(recall_es, precision_es, color = "blue", lw = 1)

plt.plot([0, 1], [1, 0], linestyle="-", color="gray")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Ensemble)")
plt.show()

plt.plot(fc_pred_es, obs_freq_es, color = "blue", lw = 1)
plt.plot([0, 1], [0, 1], color="gray", lw = 0.5)
plt.xlabel("Forecast probability")
plt.ylabel("Observed frequency")
plt.title("Reliability Diagram (Ensemble)")
plt.show()

plt.plot(far_es, hr_es, color = "blue", lw=1, label="AROC = %0.5f" % aroc_es)
plt.plot([-0.01, 1.01], [-0.01, 1.01], linestyle="-", color="gray")  
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Alarm Rate")
plt.ylabel("Hit Rate")
plt.title("ROC Curve (Ensemble)")
plt.legend(loc="lower right")
plt.show()