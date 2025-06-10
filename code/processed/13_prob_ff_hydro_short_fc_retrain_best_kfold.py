import os
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
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

# Runtime: ~ 10 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# eval_matric (string): evaluation metric used for hyperparameter tuning.
# model_2_train_list (list of strings): names of the considered machine learning models. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# kfold_count (integer): count of outter k-folds considering during the hyperparameters tunning.
# git_repo (string): repository's local path.
# file_in_train (string): relative path of the file containing the training dataset.
# file_in_test (string): relative path of the file containing the test dataset.
# dir_in_model (string): relative path of the directory containing the machine learning models to consider.
# dir_out (string): relative path of the directory containing the final version of the trained machine learning models.

##################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
eval_matric = "auprc"
kfold_count = 5
model_2_train_list = ["feed_forward_keras"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_train = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
file_in_test  = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
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


#################
# MODEL REGISTRY #
#################

def build_feed_forward_keras(template_path: str, input_dim: int):
      tmp_model = load_model(template_path) # loads the model with weights, architectures, and hyperparameters
      opt_cfg   = tmp_model.optimizer.get_config() # imports the optimised parameters
      new_opt   = type(tmp_model.optimizer).from_config(opt_cfg)

      model = load_model(template_path, compile=False) # loads the model with weights, architectures to retrain fresh over the full training dataset
      model.compile(
            optimizer=new_opt, # imports Optuna's optimised hyperparameters
            loss="sparse_categorical_crossentropy",
            metrics=[],
            run_eagerly=True
            )

      return model

MODEL_MAP = {
      "random_forest_xgboost": XGBRFClassifier,
      "random_forest_lightgbm": LGBMClassifier,
      "gradient_boosting_xgboost": XGBClassifier,
      "gradient_boosting_lightgbm": LGBMClassifier,
      "gradient_boosting_catboost": CatBoostClassifier,
      "gradient_boosting_adaboost": AdaBoostClassifier,
}


#######################
# TRAIN THE FINAL MODEL #
#######################

def build_final_model(model_name: str,
                      fold_model_file: str,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      dir_out: str):
    
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
            
            scaler = StandardScaler().fit(X_train) # Standardize the inputs
            X_train_proc = scaler.transform(X_train)
            X_test_proc  = scaler.transform(X_test)

            final_model = build_feed_forward_keras(fold_model_file, input_dim=X_train_proc.shape[1])

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
                  verbose=0
            )

            final_model.save(os.path.join(dir_out, "model.h5")) # model
            joblib.dump(scaler, os.path.join(dir_out, "scaler.joblib")) # scaling parameters used during training to use again when deploying or re-running the model

            def _predict(m, X_): # helper to create the predictions
                  return m.predict(X_)[:, 1] # we slice only the samples that belong to the positive class (i.e. yes flash flood events)
            
            y_pred_prob_train = _predict(final_model, X_train_proc)
            y_pred_prob_test  = _predict(final_model, X_test_proc)
            
      else: # decision-tree-based models

            best_fold_model = joblib.load(fold_model_file)
            best_params = best_fold_model.get_params(deep=False) # re-train on the full dataset with the same hyper-parameters
            best_params.pop("use_label_encoder", None)
            best_params.pop("algorithm", None)
            model_class = MODEL_MAP[model_name]
            final_model = model_class(**best_params) # fresh estimator
            final_model.fit(X_train, y_train)
            joblib.dump(final_model, os.path.join(dir_out, "model.joblib"))

            y_pred_prob_train = final_model.predict_proba(X_train)[:, 1]
            y_pred_prob_test  = final_model.predict_proba(X_test)[:, 1]

      for split_name, y_true, y_prob in [
            ("train", y_train, y_pred_prob_train),
            ("test",  y_test,  y_pred_prob_test)
      ]:
            precision, recall, thr = precision_recall_curve(y_true, y_prob)
            f1_curve = 2 * (precision * recall) / (precision + recall + 1e-6)
            best_thr = thr[np.argmax(f1_curve)] if thr.size else 0.5
            y_pred = (y_prob >= best_thr).astype(int)

            obs_freq, fc_pred = calibration_curve(y_true, y_prob, n_bins=100)
            far, hr, thr_roc = roc_curve(y_true, y_prob)
            aroc = auc(far, hr)
            auprc = average_precision_score(y_true, y_prob)
            fb = y_pred.sum() / max(y_true[y_pred == 1].sum(), 1)

            np.save(os.path.join(dir_out, f"obs_freq_{split_name}.npy"), obs_freq)
            np.save(os.path.join(dir_out, f"prob_pred_{split_name}.npy"), fc_pred)
            np.save(os.path.join(dir_out, f"far_{split_name}.npy"), far)
            np.save(os.path.join(dir_out, f"hr_{split_name}.npy"), hr)
            np.save(os.path.join(dir_out, f"thr_roc_{split_name}.npy"), thr_roc)
            np.save(os.path.join(dir_out, f"precision_{split_name}.npy"), np.array(precision))
            np.save(os.path.join(dir_out, f"recall_{split_name}.npy"), np.array(recall))
            np.save(os.path.join(dir_out, f"aroc_{split_name}.npy"), np.array(aroc))
            np.save(os.path.join(dir_out, f"auprc_{split_name}.npy"), np.array(auprc))
            np.save(os.path.join(dir_out, f"fb_{split_name}.npy"), np.array(fb))
            np.save(os.path.join(dir_out, f"best_thr_{split_name}.npy"), np.array(best_thr))

##############################################################################


# Read the training and the test datasets
logger.info(f"\n\nReading the training dataset")
file_in_train = git_repo + "/" + file_in_train
file_in_test = git_repo + "/" + file_in_test
X_train, y_train = load_data(file_in_train, feature_cols, target_col)
X_test, y_test = load_data(file_in_test, feature_cols, target_col)

# Uploading the trained model
for ind_model in range(len(model_2_train_list)):

      model_2_train = model_2_train_list[ind_model]
      
      # Train the final version of the model
      if model_2_train == "feed_forward_keras":
            model_ext = ".h5"
      else:
            model_ext = ".joblib"

      for k_fold in range(kfold_count):

            logger.info(f"Re-training the {model_2_train} model - k-fold n. {k_fold + 1}")

            # Create the output directory
            dir_out_temp = git_repo + "/" + dir_out + "/" + eval_matric + "/" + model_2_train + "/fold" + str(k_fold + 1) 
            os.makedirs(dir_out_temp, exist_ok=True)

            file_in = git_repo + "/" + dir_in_model + "/" + eval_matric + "/" + model_2_train + "/model_rep1_fold" + str(k_fold + 1) + model_ext
            build_final_model(model_2_train, file_in, X_train, y_train, X_test, y_test, dir_out_temp)