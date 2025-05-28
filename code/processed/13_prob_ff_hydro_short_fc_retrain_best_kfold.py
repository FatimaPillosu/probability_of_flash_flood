import os
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

##################################################################################################
# CODE DESCRIPTION
# 13_prob_ff_hydro_short_fc_retrain_best_kfold.py retrains the best model from k-fold cross validation over the whole  
# training dataset, and tests it over the test data.

# Usage: python3 13_prob_ff_hydro_short_fc_retrain_best_kfold.py

# Runtime: ~ 2 hours.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

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
# best_kfold_list (list of integers): list of kfolds containing the best set of hyperparameters for the correspondent model.
# git_repo (string): repository's local path.
# file_in_train (string): relative path of the file containing the training dataset.
# file_in_test (string): relative path of the file containing the test dataset.
# dir_in_model (string): relative path of the directory containing the machine learning models to consider.
# dir_out (string): relative path of the directory containing the final version of the trained machine learning models.

##################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_50", "swvl", "sdfor", "lai"]
target_col = "ff"
model_2_train_list = ["feed_forward_keras", "gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost", "random_forest_xgboost", "random_forest_lightgbm", ]
best_kfold_list = [3, 1, 3, 1, 1, 1]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_train = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
file_in_test = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2023.csv"
dir_in_model = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
##################################################################################################


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


#######################
# TRAIN THE FINAL MODEL #
#######################

MODEL_MAP = {
      "random_forest_xgboost": XGBRFClassifier,
      "random_forest_lightgbm": LGBMClassifier,
      "gradient_boosting_xgboost": XGBClassifier,
      "gradient_boosting_lightgbm": LGBMClassifier,
      "gradient_boosting_catboost": CatBoostClassifier,
      "gradient_boosting_adaboost": AdaBoostClassifier,
}

def build_final_model(model_name, fold_model_file, X_train, y_train, X_test, y_test, dir_out):
    
      """
      WORKFLOW:
      1. Load the best model from a given fold (by file path).
      2. Extract its hyperparameters.
      3. Create a fresh instance of the chosen model class with those hyperparameters.
      4. Train the model on the entire training dataset.
      5. Apply the model over the test dataset.
      
      PARAMETERS:
      model_name : str
            A key to pick which model class to use. Must be in MODEL_MAP.
      fold_model_file : str
            Path to the .joblib file (joblib dump) of the best fold's model.
      X_train : np.array or pd.DataFrame
            Input features for the entire training dataset.
      y_train : np.array or pd.Series
            Target labels for the entire training dataset.
      X_test : np.array or pd.DataFrame
            Input features for the test dataset.
      y_test : np.array or pd.Series
            Target labels for the test dataset.
      
      Returns
      -------
      final_model
            Trained model on the full dataset, using the best fold's hyperparameters.
      """

      # Load the model from joblib and extarct the model parameters
      if model_name == "feed_forward_keras":
            best_fold_model = load_model(fold_model_file)
      else:
            best_fold_model = joblib.load(fold_model_file)
            best_params = best_fold_model.get_params()
            if 'use_label_encoder' in best_params:
                  del best_params['use_label_encoder']

      # Retrieve the model class from MODEL_MAP
      model_class = MODEL_MAP.get(model_name)

      # Create a fresh instance of the chosen model with the best hyperparams
      if model_name == "feed_forward_keras":
            final_model = best_fold_model
      else:
            final_model = model_class(**best_params)

      # Retrain the model over the whole training dataset, and save it
      final_model.fit(X_train, y_train)

      if model_name == 'feed_forward_keras':
            final_model.save(os.path.join(dir_out, "model.h5"))
      else:
            joblib.dump(final_model, os.path.join(dir_out, "model.joblib"))


      # Evaluate the model over the training dataset, and save the scores
      X = X_train
      y = y_train
      name = "train"
      if model_name == 'feed_forward_keras':
            y_pred_prob = final_model.predict(X.values)[:, 1]
      else:
            y_pred_prob = final_model.predict_proba(X)[:, 1]

      precision, recall_curve, thresholds = precision_recall_curve(y, y_pred_prob)
      f1_curve = 2 * (precision * recall_curve) / (precision + recall_curve + 1e-6)
      best_idx = np.argmax(f1_curve)
      best_threshold = thresholds[best_idx]
      y_pred = (y_pred_prob >= best_threshold).astype(int)

      obs_freq, fc_pred = calibration_curve(y, y_pred_prob, n_bins=100)
      far, hr, thr_roc = roc_curve(y, y_pred_prob)
      aroc = auc(far, hr)
      recall = recall_score(y, y_pred)
      f1 = f1_score(y, y_pred)

      test_scores = np.array([recall, f1, aroc, best_threshold])
      np.save(dir_out + "/obs_freq_" + name, obs_freq)
      np.save(dir_out + "/fc_pred_" + name, fc_pred)
      np.save(dir_out + "/far_" + name, far)
      np.save(dir_out + "/hr_" + name, hr)
      np.save(dir_out + "/thr_roc_" + name, thr_roc)
      np.save(dir_out + "/test_scores_" + name, test_scores)

      # Evaluate the model over the test dataset, and save the scores
      X = X_test
      y = y_test
      name = "test"
      if model_name == 'feed_forward_keras':
            y_pred_prob = final_model.predict(X.values)[:, 1]
      else:
            y_pred_prob = final_model.predict_proba(X)[:, 1]

      precision, recall_curve, thresholds = precision_recall_curve(y, y_pred_prob)
      f1_curve = 2 * (precision * recall_curve) / (precision + recall_curve + 1e-6)
      best_idx = np.argmax(f1_curve)
      best_threshold = thresholds[best_idx]
      y_pred = (y_pred_prob >= best_threshold).astype(int)

      obs_freq, fc_pred = calibration_curve(y, y_pred_prob, n_bins=100)
      far, hr, thr_roc = roc_curve(y, y_pred_prob)
      aroc = auc(far, hr)
      recall = recall_score(y, y_pred)
      f1 = f1_score(y, y_pred)

      test_scores = np.array([recall, f1, aroc, best_threshold])
      np.save(dir_out + "/obs_freq_" + name, obs_freq)
      np.save(dir_out + "/fc_pred_" + name, fc_pred)
      np.save(dir_out + "/far_" + name, far)
      np.save(dir_out + "/hr_" + name, hr)
      np.save(dir_out + "/thr_roc_" + name, thr_roc)
      np.save(dir_out + "/test_scores_" + name, test_scores)

##############################################################################


# Read the training and the test datasets
logger.info(f"\n\nReading the training dataset")
file_in_train = git_repo + "/" + file_in_train
X_train, y_train = load_data(file_in_train, feature_cols, target_col)

logger.info(f"\n\nReading the test dataset")
file_in_test = git_repo + "/" + file_in_test
X_test, y_test = load_data(file_in_test, feature_cols, target_col)


# Uploading the trained model
for ind_model in range(len(model_2_train_list)):

      model_2_train = model_2_train_list[ind_model]
      kfold = best_kfold_list[ind_model]
      logger.info(f"Re-training the {model_2_train} model")
      
      # Create the output directory
      dir_out_temp = git_repo + "/" + dir_out + "/" + model_2_train
      os.makedirs(dir_out_temp, exist_ok=True)

      # Train the final version of the model
      if model_2_train == "feed_forward_keras":
            model_ext = ".h5"
      else:
            model_ext = ".joblib"
      file_in = git_repo + "/" + dir_in_model + "/" + model_2_train + "/model_" + str(kfold) + model_ext
      build_final_model(model_2_train, file_in, X_train, y_train, X_test, y_test, dir_out_temp)