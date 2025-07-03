import os
import sys
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow import keras
from tensorflow.keras.models import load_model
import joblib
import json
from sklearn.metrics import precision_recall_curve

#############################################################################################
# CODE DESCRIPTION
# 13_prob_ff_hydro_short_fc_retrain_best_kfold.py retrains the best model from k-fold cross validation over the 
# whole training dataset, and tests it over the test data.

# Usage: python3 13_prob_ff_hydro_short_fc_retrain_best_kfold.py

# Runtime: ~ 5 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# model_2_train (string): name of the model to train. Valid values are:
#                                               - random_forest_xgboost
#                                               - random_forest_lightgbm
#                                               - gradient_boosting_xgboost
#                                               - gradient_boosting_lightgbm
#                                               - gradient_boosting_catboost
#                                                - feed_forward_keras
# loss_fn_choice (string): type of loss function considered. Valid values are:
#                                               - bce: no weights applied to loss function.
#                                               - weighted_bce: wheight applied to loss function.
# eval_metric (string): evaluation metric for the data-driven models. Valid values are:
#                                         - auc: area under the roc curve.
#                                         - auprc: area under the precion-recall curve.
# rep_to_run (positive integer): repetition from the nested cross-validation to run.
# outer_fold_to_run (positive integer): outer fold from the nested cross-validation to run.
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# git_repo (string): repository's local path.
# file_in_train (string): relative path of the file containing the training dataset.
# file_in_test (string): relative path of the file containing the test dataset.
# dir_in (string): relative path of the directory containing the machine learning model to consider.
# dir_out (string): relative path of the directory containing the final version of the trained machine learning model.

#############################################################################################
# INPUT PARAMETERS
model_2_train = sys.argv[1]
loss_fn_choice = sys.argv[2]
eval_metric = sys.argv[3]
rep_to_run = int(sys.argv[4])
outer_fold_to_run = int(sys.argv[5])
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_train = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
file_in_test  = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
dir_in = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
dir_out = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
##################################################################################################


#########################
# CHECK INPUT PARAMETERS #
#########################

valid_models = {"random_forest_xgboost", "random_forest_lightgbm", "gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost", "feed_forward_keras"}
assert model_2_train in valid_models, f"Invalid model: {model_2_train}"
assert eval_metric in {'auc', 'auprc'}, f"Invalid eval_metric: {eval_metric}"
assert loss_fn_choice in {'bce', 'weighted_bce'}, f"Invalid loss function: {loss_fn_choice}"


#################################
# SET-UP APPLICATION-WIDE LOGGING #
#################################

logging.basicConfig(
      level=logging.INFO, # records only messages at INFO level or higher (INFO, WARNING, ERROR, CRITICAL).
      format="%(asctime)s [%(levelname)s] %(name)s - %(message)s" # defines how each log line will look.
      )
logger = logging.getLogger(__name__) # object for emitting log records.


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
# METRICS UTILITIES #
##################

def get_keras_metric(metric_name: str):
      if metric_name == 'auc':
            return keras.metrics.AUC(name='auc')
      elif metric_name == 'auprc':
            return keras.metrics.AUC(name='auprc', curve='PR')
      else:
            raise ValueError(f"Unsupported metric: {metric_name}")


#################
# MODEL REGISTRY #
#################

def build_keras_model(template_path: str, input_dim: int, y_train, metric_name: str, loss_fn_choice="bce"):
      
      tmp_model = load_model(template_path) # loads the model with weights, architectures, and hyperparameters
      opt_cfg   = tmp_model.optimizer.get_config() # imports the optimised parameters
      new_opt   = type(tmp_model.optimizer).from_config(opt_cfg)
      model = load_model(template_path, compile=False) # loads the model with weights, architectures to retrain fresh over the full training dataset

      model.compile(
            optimizer = new_opt, # imports Optuna's optimised hyperparameters
            loss = tmp_model.loss,
            metrics = [get_keras_metric(metric_name)],
            run_eagerly = True
            )
      
      return model

MODEL_MAP = {
      'feed_forward_keras': build_keras_model,
      "random_forest_xgboost": XGBRFClassifier,
      "random_forest_lightgbm": LGBMClassifier,
      "gradient_boosting_xgboost": XGBClassifier,
      "gradient_boosting_lightgbm": LGBMClassifier,
      "gradient_boosting_catboost": CatBoostClassifier,
}


#######################
# TRAIN THE FINAL MODEL #
#######################

def build_final_model(
      X_train: pd.DataFrame,
      y_train: pd.Series,
      X_test: pd.DataFrame,
      y_test: pd.Series,
      model_name: str,
      fold_model_file: str,
      metric_name: str,
      loss_fn_choice: str,
      pos_weight: float,
      dir_out: str
      ):
    
      """
      WORKFLOW:
      1. Load the best models for each outer fold.
      2. Extract its hyperparameters.
      3. Create a fresh instance of the chosen model class with those hyperparameters.
      4. Train the model on the entire training dataset.
      
      PARAMETERS:
      model_name : str
            A key to pick which model class to use. Must be in MODEL_MAP.
      fold_model_file : str
            Path to the .joblib file (joblib dump) of the best fold's model.
      X_train : np.array or pd.DataFrame
            Input features for the entire training dataset.
      y_train : np.array or pd.Series
            Target labels for the entire training dataset.
      
      Returns
      -------
      final_model
            Trained model on the full dataset, using the best fold's hyperparameters.
      """

      # Retrain the models
      if model_name == "feed_forward_keras": # feed-forward neural network
            
            scaler = StandardScaler().fit(X_train) 
            X_train_proc = scaler.transform(X_train)
            X_test_proc  = scaler.transform(X_test)

            final_model = build_keras_model(fold_model_file, input_dim=X_train_proc.shape[1], y_train=y_train, metric_name = metric_name, loss_fn_choice = loss_fn_choice)

            es = keras.callbacks.EarlyStopping(
                  monitor="val_loss",
                  patience=2,
                  restore_best_weights=True
            )

            final_model.fit(
                  X_train_proc,
                  y_train.values,
                  validation_split=0.1,
                  batch_size=64,
                  epochs=20,
                  callbacks=[es],
                  class_weight={0: 1.0, 1: pos_weight},
                  verbose=0
            )

            final_model.save(os.path.join(dir_out, "model.h5")) 
            joblib.dump(scaler, os.path.join(dir_out, "scaler.joblib"))

            def _predict(m, X_): 
                  return m.predict(X_).flatten()
            
            y_pred_prob_train = _predict(final_model, X_train_proc)
            y_pred_prob_test  = _predict(final_model, X_test_proc)
            
      else: # decision-tree-based models

            best_fold_model = joblib.load(fold_model_file)
            best_params = best_fold_model.get_params(deep=False)
            best_params.pop("use_label_encoder", None)
            best_params.pop("algorithm", None)
            model_class = MODEL_MAP[model_name]
            final_model = model_class(**best_params)
            final_model.fit(X_train, y_train)
            joblib.dump(final_model, os.path.join(dir_out, "model.joblib"))

            y_pred_prob_train = final_model.predict_proba(X_train)[:, 1]
            y_pred_prob_test  = final_model.predict_proba(X_test)[:, 1]

      # Find best threshold to convert continuous probabilities into binary classes
      precision, recall, thr_pr = precision_recall_curve(y_train, y_pred_prob_train)
      f1_curve = 2 * (precision * recall) / (precision + recall + 1e-6)
      best_thr = thr_pr[np.argmax(f1_curve)] if thr_pr.size else 0.5

      np.save(os.path.join(dir_out, f"obs_train.npy"), y_train)
      np.save(os.path.join(dir_out, f"obs_test.npy"), y_test)
      np.save(os.path.join(dir_out, f"fc_train.npy"), y_pred_prob_train)
      np.save(os.path.join(dir_out, f"fc_test.npy"), y_pred_prob_test)
      np.save(os.path.join(dir_out, f"best_thr.npy"), best_thr)

##############################################################################


# Read the training and the test datasets
logger.info(f"\n\nReading the training dataset")
file_in_train = git_repo + "/" + file_in_train
file_in_test = git_repo + "/" + file_in_test
X_train, y_train = load_data(file_in_train, feature_cols, target_col)
X_test, y_test = load_data(file_in_test, feature_cols, target_col)
# Train the considered machine learning model
logger.info(f"Re-training the {model_2_train} model - k-fold n. {outer_fold_to_run}")

if model_2_train == "feed_forward_keras":
      model_ext = "h5"
else:
      model_ext = "joblib"

file_in = f"{git_repo}/{dir_in}/{loss_fn_choice}/{eval_metric}/{model_2_train}/model_rep{rep_to_run}_fold{outer_fold_to_run}.{model_ext}"

best_params_file = f"{git_repo}/{dir_in}/{loss_fn_choice}/{eval_metric}/{model_2_train}/optuna/best_params_rep{rep_to_run}_fold{outer_fold_to_run}.json"
with open(best_params_file, "r") as f:
    data = json.load(f)
pos_weight = data["loss_fn_details"]["pos_weight"]

dir_out_temp = f"{git_repo}/{dir_out}/{loss_fn_choice}/{eval_metric}/{model_2_train}"
os.makedirs(dir_out_temp, exist_ok=True)

build_final_model(
      X_train, 
      y_train, 
      X_test, 
      y_test, 
      model_2_train,
      file_in,
      eval_metric,
      loss_fn_choice,
      pos_weight,
      dir_out_temp
      )