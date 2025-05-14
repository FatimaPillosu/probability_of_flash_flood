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
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

##############################################################################################
# CODE DESCRIPTION
# 25_compute_retrain_ml_test.py retrains the best model from k-fold cross validation over the whole training 
# dataset, and tests it over the test data.
# Runtime: the code takes up to 2 hours.

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
ensemble_model_list = ["gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost", "random_forest_xgboost", "random_forest_lightgbm"]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_train = "data/compute/21_combine_pdt/pdt_2001_2020.csv"
file_in_test = "data/compute/21_combine_pdt/pdt_2024_2024.csv"
dir_in = "data/compute/25_retrain_ml_test"
dir_out = "data/compute/27_ensemble_stacking"
##############################################################################################


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
      "feed_forward_keras": AdaBoostClassifier
}

def build_final_model(model_name, fold_model_file, X_train):
    
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
      model = joblib.load(fold_model_file)
      
      # Evaluate the model over 
      if model_name == 'feed_forward_keras':
            y_pred_prob = model.predict(X_train.values)[:, 1]
      else:
            y_pred_prob = model.predict_proba(X_train)[:, 1]

      return y_pred_prob


###################################
# OPTIMIZE META_MODEL WITH OPTUNA #
###################################

def optimize_meta_logistic(train_ensemble, y_train, n_trials=50, cv_folds=5):
    
      """
      Optimize a Logistic Regression meta-model using Optuna by 
      maximizing the F1 score, including an optimization of the 
      classification threshold for each CV fold.
      
      For each trial:
            1. A LogisticRegression model is instantiated with hyperparameters suggested by Optuna.
            2. The model is evaluated using stratified cross-validation.
            3. For each fold, the best threshold is computed using the precision_recall_curve to maximize the F1 score.
            4. The mean F1 score across folds is returned as the trial's objective.

      PARAMETERS
      train_ensemble : pd.DataFrame or np.array
            Stacked predictions from the base models (features for the meta-learner).
      y_train : pd.Series or np.array
            True target values for training.
      n_trials : int, optional
            Number of Optuna trials to run. Default is 50.
      cv_folds : int, optional
            Number of cross-validation folds. Default is 5.
      
      RETURNS
      best_params : dict
            The best hyperparameters found for LogisticRegression.
      best_score : float
            The best cross-validated F1 score achieved.
      """

      def objective(trial):

            # Define the hyperparameter search space
            C = trial.suggest_loguniform("C", 1e-4, 1e2)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            
            # Use 'saga' solver which supports both l1 and l2 penalties
            solver = "saga"
            max_iter = 1000

            # Initialize the logistic regression model with trial hyperparameters
            model = LogisticRegression(
                  C=C,
                  penalty=penalty,
                  solver=solver,
                  max_iter=max_iter,
                  random_state=42
            )
            
            # Define a stratified k-fold cross-validation.
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            f1_scores = []

            # Iterate through the CV folds
            for train_idx, val_idx in cv.split(train_ensemble, y_train):
            
                  X_train_cv = train_ensemble.iloc[train_idx] if hasattr(train_ensemble, "iloc") else train_ensemble[train_idx]
                  y_train_cv = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
                  X_val_cv = train_ensemble.iloc[val_idx] if hasattr(train_ensemble, "iloc") else train_ensemble[val_idx]
                  y_val_cv = y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]

                  model.fit(X_train_cv, y_train_cv)
                  y_val_prob = model.predict_proba(X_val_cv)[:, 1]

                  # Compute precision-recall curve and optimize threshold for F1 score
                  precision, recall_array, thresholds = precision_recall_curve(y_val_cv, y_val_prob)
                  f1_curve = 2 * (precision * recall_array) / (precision + recall_array + 1e-6)
                  if len(thresholds) > 0:
                        best_idx = np.argmax(f1_curve)
                        best_thr = thresholds[best_idx]
                  else:
                        best_thr = 0.5

                  y_val_pred = (y_val_prob >= best_thr).astype(int)
                  f1_val = f1_score(y_val_cv, y_val_pred)
                  f1_scores.append(f1_val)

            return np.mean(f1_scores) # return the average F1 score over all folds as the objective

      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=n_trials)

      best_params = study.best_trial.params
      best_score = study.best_trial.value
      
      return best_params, best_score

##############################################################################


# Read the training and the test datasets
print(f"Reading the training dataset")
file_in_train = git_repo + "/" + file_in_train
X_train, y_train = load_data(file_in_train, feature_cols, target_col)

print(f"Reading the test dataset")
file_in_test = git_repo + "/" + file_in_test
X_test, y_test = load_data(file_in_test, feature_cols, target_col)

# Creating the base models for the ensemble stacking
train_ensemble = pd.DataFrame()
test_ensemble = pd.DataFrame()
for ensemble_model in ensemble_model_list:
      file_model_in = git_repo + "/" + dir_in + "/" + ensemble_model + "/model.joblib"
      train_ensemble[ensemble_model] = build_final_model(ensemble_model, file_model_in, X_train)
      test_ensemble[ensemble_model] = build_final_model(ensemble_model, file_model_in, X_test)

# Creating the meta-model for the ensemble stacking
best_params, best_f1 = optimize_meta_logistic(train_ensemble, y_train, n_trials=50, cv_folds=5)
print("Best Hyperparameters:", best_params)
print("Best F1 Score:", best_f1)
meta_model = LogisticRegression(**best_params, solver="saga", max_iter=1000, random_state=42)
meta_model.fit(train_ensemble, y_train)
y_pred_prob = meta_model.predict_proba(test_ensemble.values)[:, 1]

# Threshold tuning for best F1
precision, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_curve = 2 * (precision * recall_curve) / (precision + recall_curve + 1e-6)
best_idx = np.argmax(f1_curve)
best_threshold = thresholds[best_idx]
y_pred = (y_pred_prob >= thresholds[best_idx]).astype(int)

# Testing the model
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