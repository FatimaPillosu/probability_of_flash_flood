import os
import sys
import logging
import inspect
from typing import List, Tuple
import json
import numpy as np
import pandas as pd
import optuna
from optuna.integration import (
    TFKerasPruningCallback,
    XGBoostPruningCallback,
    LightGBMPruningCallback,
    CatBoostPruningCallback,
)
from optuna.trial import FixedTrial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from packaging import version
from tensorflow import keras
import joblib


##################################################################################################
# CODE DESCRIPTION
# 12_prob_ff_hydro_short_fc_train_ml_cv_optuna.py trains the considered machine learning models (i.e., decision trees  
# and neural networks) to forecast the probabilities of flash flood. The training applies a nested stratified k-fold 
# cross-validation technique to train with imbalanceee datasets, and optuna for hyperparameter tuning.

# Usage: python3 12_prob_ff_hydro_short_fc_train_ml_cv_optuna.py

# Runtime: ~ 3 days per model.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# model_2_train_list (list of strings): names of the models to train. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - gradient_boosting_adaboost
#                                                                 - feed_forward_keras
# git_repo (string): repository's local path.
# file_in (string): relative path of the file containing the training dataset.
# dir_out (string): relative path of the directory containing the trained machine learning models.

##################################################################################################
# INPUT PARAMETERS
model_2_train = sys.argv[1]
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2001_2020.csv"
dir_out = "data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna"
##################################################################################################


#################################
# SET-UP APPLICATION-WIDE LOGGING #
#################################

logging.basicConfig(
      level=logging.INFO, # records only messages at INFO level or higher (INFO, WARNING, ERROR, CRITICAL).
      format="%(asctime)s [%(levelname)s] %(name)s - %(message)s" # defines how each log line will look.
      )
logger = logging.getLogger(__name__) # object for emitting log records.
optuna.logging.set_verbosity(optuna.logging.INFO)


##########################
#GLOBAL PRUNER DEFINITION #
##########################

is_new_xgb = version.parse(xgb.__version__) >= version.parse("1.6.0")
PRUNER = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
HAS_XGB_CALLBACK = "callbacks" in inspect.signature(XGBClassifier.fit).parameters


###############
# DATA LOADING #
###############

def load_data(
            csv_path: str, 
            feature_cols: List[str], 
            target_col: str
            ) -> Tuple[pd.DataFrame, pd.Series]:
      
      """
      Read the point data table file containing the data for the values for the target variable and the features, and split it into feature matrix "X" and target vector "y".

      Input parameters:
            csv_path : str
                  Absolute or relative path to the input CSV file.
            feature_cols : list[str]
                  Column names to be used as predictors.
            target_col : str
                  Column name to be used as the predictand.

      Output(s):
            Tuple[pandas.DataFrame, pandas.Series]
                  "X" : a DataFrame containing only "feature_cols";  
                  "y" : a Series containing the "target_col" values.
      """
      
      logger.info(f"Loading dataset from {csv_path}") # log where the CSV is being read from
      df = pd.read_csv(csv_path) # read entire CSV into a DataFrame
      X = df[feature_cols].copy() # extract feature matrix X (deep copy to avoid SettingWithCopy warnings)
      y = df[target_col].copy() # # extract target vector y
      logger.info(f"Loaded dataset with shape X={X.shape}, y={y.shape}") # log shapes for quick sanity-check
      return X, y


#################
# MODEL REGISTRY #
#################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lgb_auprc(preds, dataset):
    y_true = dataset.get_label()
    return "auprc", average_precision_score(y_true, sigmoid(preds)), True

def build_keras_model(trial, input_dim: int):
      n_layers = trial.suggest_int("n_layers", 1, 3)
      model = keras.Sequential()
      model.add(keras.Input(shape=(input_dim,)))
      for i in range(n_layers):
            num_units = trial.suggest_int(f"units_{i}", 32, 256)
            model.add(keras.layers.Dense(num_units, activation='relu'))
            dropout_rate = trial.suggest_float(f"dropout_{i}", 0.0, 0.5)
            model.add(keras.layers.Dropout(dropout_rate))
      model.add(keras.layers.Dense(2, activation='softmax'))
      lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
      opt = keras.optimizers.Adam(learning_rate=lr)
      model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[])
      return model

############################################
def build_feed_forward_keras(trial, input_dim=None):
    return build_keras_model(trial, input_dim=input_dim)

#################################
def build_xgb_rf(trial, *_):
      params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
      }
      return XGBRFClassifier(**params)

##################################
def build_xgb_gb(trial, *_):
      params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42
      }
      return XGBClassifier(**params)

#################################
def build_lgb_rf(trial, *_):
      params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'boosting_type': 'rf',
            'objective': 'binary',
            'eval_metric': 'average_precision',
            'random_state': 42
      }
      return LGBMClassifier(**params)

#################################
def build_lgb_gb(trial, *_):
      params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'eval_metric': 'average_precision',
            'random_state': 42
      }
      return LGBMClassifier(**params)

###################################
def build_catboost(trial, *_):
      params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
            'loss_function': 'Logloss',
            'eval_metric': 'PRAUC:type=Classic',
            'verbose': 0,
            'random_state': 42
      }
      return CatBoostClassifier(**params)

######################################
def build_adaboost_gb(trial, *_):
      
      stump = DecisionTreeClassifier(
            max_depth=trial.suggest_int("dt_max_depth", 1, 3),
            min_samples_leaf=trial.suggest_int("dt_min_samples_leaf", 1, 10),
            )

      return AdaBoostClassifier(
            estimator=stump,              
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0),
            algorithm="SAMME",
            random_state=42
            )

##################
MODEL_REGISTRY = {
      'feed_forward_keras': build_feed_forward_keras,
      'random_forest_xgboost': build_xgb_rf,
      'gradient_boosting_xgboost': build_xgb_gb,
      'random_forest_lightgbm': build_lgb_rf,
      'gradient_boosting_lightgbm': build_lgb_gb,
      'gradient_boosting_catboost': build_catboost,
      'gradient_boosting_adaboost': build_adaboost_gb
      }

###################################################
def build_model(model_type: str, trial, input_dim: int = None):
      if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type: {model_type}")
      build_func = MODEL_REGISTRY[model_type]
      return build_func(trial, input_dim)


##########################
# NESTED CV + OPTUNA LOGIC #
##########################

def evaluate_auc(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    return auc(fpr, tpr)

def evaluate_auprc(y_true, y_pred_prob):
    return average_precision_score(y_true, y_pred_prob)

################################
def _make_dir_if_not_exists(path: str):
      os.makedirs(path, exist_ok=True)

##################################################################
def inner_objective(trial, model_type, X_train, y_train, n_splits=5, n_repeats=10):

      logger.info(f"   · Trial {trial.number} started") 

      # --- Neural Network Models --- #
      if model_type == 'feed_forward_keras':
            
            kf = RepeatedStratifiedKFold(
                  n_splits=n_splits, 
                  n_repeats=n_repeats,
                  random_state=42)
            
            auprc_scores = []

            for train_idx, val_idx in kf.split(X_train, y_train):

                  X_trf, X_valf = X_train.iloc[train_idx], X_train.iloc[val_idx]
                  y_trf, y_valf = y_train.iloc[train_idx], y_train.iloc[val_idx]
                  
                  scaler = StandardScaler().fit(X_trf) # standardisation of the features and target variables
                  X_trf = scaler.transform(X_trf)
                  X_valf = scaler.transform(X_valf)

                  model = build_model(model_type, trial, input_dim = X_trf.shape[1])

                  early_stop = keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=2,
                        restore_best_weights=True
                        )
                  
                  pruning_cb = TFKerasPruningCallback(trial, 'val_loss')

                  model.fit(
                        X_trf, 
                        y_trf.values,
                        batch_size=trial.suggest_int("batch_size", 32, 128),
                        epochs=20,
                        validation_data=(X_valf, y_valf.values),
                        callbacks=[early_stop, pruning_cb],
                        verbose=0
                        )

                  y_val_prob = model.predict(X_valf)[:, 1]
                  auprc_scores.append(evaluate_auprc(y_valf, y_val_prob))

            return np.mean(auprc_scores)
      
      # --- Tree-Based Models --- #
      kf = RepeatedStratifiedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats,
            random_state=42
            )
      
      auprc_scores = []

      for train_idx, val_idx in kf.split(X_train, y_train):
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = build_model(model_type, trial)

            if model_type in {'gradient_boosting_xgboost', 'random_forest_xgboost'}:
            
                  if HAS_XGB_CALLBACK:
                        model.fit(
                              X_tr,
                              y_tr,
                              eval_set=[(X_val, y_val)],
                              verbose=False,
                              callbacks=[XGBoostPruningCallback(trial, "validation_0-aucpr")],
                        )
                  else:
                        model.fit(
                              X_tr,
                              y_tr,
                              eval_set=[(X_val, y_val)],
                              verbose=False,
                        )
                 
            elif model_type in {"gradient_boosting_lightgbm", "random_forest_lightgbm"}:
                  
                  model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric="average_precision",
                        callbacks=[LightGBMPruningCallback(trial, "average_precision")],
                        )

            elif model_type == "gradient_boosting_catboost":
                  
                  model.fit(
                        X_tr,
                        y_tr,
                        eval_set=(X_val, y_val),
                        verbose=False,
                        callbacks=[CatBoostPruningCallback(trial, "PRAUC:type=Classic")],
                        )
            
            else:
                  
                  model.fit(X_tr, y_tr)

            y_val_prob = model.predict_proba(X_val)[:, 1]
            auprc_scores.append(evaluate_auprc(y_val, y_val_prob))
      
      return np.mean(auprc_scores)

def train_with_nested_cv_and_optuna(
      X: pd.DataFrame,
      y: pd.Series,
      model_type: str,
      dir_out: str,
      n_trials: int = 100,
      n_outer: int = 10,
      n_inner: int = 5,
      n_repeats: int = 5
      ):
      
      outer_cv = RepeatedStratifiedKFold(
            n_splits=n_outer, 
            n_repeats=n_repeats,
            random_state=42
            )

      shape_scores = (n_repeats, n_outer)
      outer_auprc   = np.zeros(shape_scores)
      outer_auc  = np.zeros(shape_scores)
      outer_best_thresholds = np.zeros(shape_scores)

      logger.info("Finished reading data. Entering outer CV loop…")

      for split_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y), 1):
            
            rep = (split_idx - 1) // n_outer    
            fold = (split_idx - 1) %  n_outer    
            fold_logger = logging.getLogger(f"{__name__}.rep{rep}.fold{fold}")
            fold_logger.info(f"=== Outer repeat {rep+1}/{n_repeats}, fold {fold+1}/{n_outer} ===")
            fold_logger.debug(f"Train positives: {y.iloc[train_index].sum():,} | " 
                              f"Test positives: {y.iloc[test_index].sum():,}"
                              )

            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

            fold_logger.info("Launching Optuna study…")

            study = optuna.create_study(
                  direction='maximize', 
                  sampler=optuna.samplers.TPESampler(seed=42),
                  pruner=PRUNER
                  )
            study.optimize(
                  lambda t: inner_objective(
                        t, model_type, X_train_outer, y_train_outer, n_splits=n_inner
                        ),
                  n_trials=n_trials,
                  )

            # ---- SAVE TRIAL DATA FOR PLOTS ---- #
            optuna_dir = os.path.join(dir_out, "optuna")
            _make_dir_if_not_exists(optuna_dir)

            df_trials = study.trials_dataframe()
            df_trials["wall_secs"] = (
                  df_trials["datetime_complete"] - df_trials["datetime_start"]
            ).dt.total_seconds()
            df_trials.to_csv(
                  os.path.join(optuna_dir, f"trials_rep{rep+1}_fold{fold+1}.csv"), index=False
            )
            with open(
                  os.path.join(optuna_dir, f"best_params_rep{rep+1}_fold{fold+1}.json"), "w"
            ) as fp:
                  json.dump(study.best_trial.params, fp, indent=2)
            # ----------------------------------- #

            fold_logger.info(f"   · Optuna done. Best AUPRC={study.best_value:.3f}")
            best_params = study.best_params
            fixed_trial = FixedTrial(best_params)

            if model_type == 'feed_forward_keras':
                  final_model = build_model(model_type, fixed_trial, input_dim=X_train_outer.shape[1])
                  scaler = StandardScaler().fit(X_train_outer)
                  X_train_scaled = scaler.transform(X_train_outer)
                  X_test_scaled = scaler.transform(X_test_outer)
            else:
                  final_model = build_model(model_type, fixed_trial)
                  X_train_scaled = X_train_outer
                  X_test_scaled = X_test_outer

            fold_logger.info("   · Fitting final model on outer-train subset…")

            if model_type == 'feed_forward_keras':
                  early_stop = keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=2,
                        restore_best_weights=True
                        )
                  final_model.fit(
                        X_train_scaled, 
                        y_train_outer.values,
                        batch_size=best_params.get("batch_size", 64),
                        epochs=20,
                        validation_data=(X_test_scaled, y_test_outer.values),
                        callbacks=[early_stop],
                        verbose=0
                        )
                  y_pred_prob = final_model.predict(X_test_scaled)[:, 1]
            else:
                  final_model.fit(X_train_scaled, y_train_outer)
                  y_pred_prob = final_model.predict_proba(X_test_scaled)[:, 1]

            precision, recall, thr_pr = precision_recall_curve(y_test_outer, y_pred_prob)
            f1_curve = 2 * (precision * recall) / (precision + recall + 1e-6)
            best_thr = thr_pr[np.argmax(f1_curve)] if thr_pr.size else 0.5

            outer_auprc[rep, fold] = evaluate_auprc(y_test_outer, y_pred_prob)
            outer_auc[rep, fold] = evaluate_auc(y_test_outer, y_pred_prob)
            outer_best_thresholds[rep, fold] = best_thr
            obs_freq, prob_pred = calibration_curve(y_test_outer, y_pred_prob, n_bins=100)
            far, hr, thr_roc = roc_curve(y_test_outer, y_pred_prob)

            fold_logger.info(
                  f"   · Outer test AUPRC={outer_auprc[rep, fold]:.3f} "
                  f"   · Outer test AUC={outer_auc[rep, fold]:.3f} "
                  )

            # Saving outputs
            _make_dir_if_not_exists(dir_out)
            prefix = f"rep{rep+1}_fold{fold+1}"
            if model_type == 'feed_forward_keras':
                  final_model.save(os.path.join(dir_out, "model_"+ str(prefix) + ".h5"))
            else:
                  joblib.dump(final_model, os.path.join(dir_out, "model_"+ str(prefix) + ".joblib"))
            np.save(os.path.join(dir_out, f"obs_freq_{prefix}.npy"), np.array(obs_freq))
            np.save(os.path.join(dir_out, f"prob_pred_{prefix}.npy"), np.array(prob_pred))
            np.save(os.path.join(dir_out, f"far_{prefix}.npy"), np.array(far))
            np.save(os.path.join(dir_out, f"hr_{prefix}.npy"), np.array(hr))
            np.save(os.path.join(dir_out, f"thr_roc_{prefix}.npy"), np.array(thr_roc))
            np.save(os.path.join(dir_out, f"precision_{prefix}.npy"), np.array(precision))
            np.save(os.path.join(dir_out, f"recall_{prefix}.npy"), np.array(recall))
      np.save(os.path.join(dir_out, "aroc"), outer_auc)
      np.save(os.path.join(dir_out, "auprc"), outer_auprc)
      np.save(os.path.join(dir_out, "best_threshold"), outer_best_thresholds)

      logger.info("★ All outer folds finished.")
      logger.info(f"Overall mean AUPRC={outer_auprc.mean():.3f} ± {outer_auc.std():.3f}")
      logger.info(f"Overall mean AUC={outer_auc.mean():.3f} ± {outer_auc.std():.3f}")

##############################################################################


# Read the training dataset (point data table)
logger.info(f"\n\nReading the training dataset")
file_in_pdt = os.path.join(git_repo, file_in)
X, y = load_data(file_in_pdt, feature_cols, target_col)

# Reduce the training dataset for faster training, while maintaing the ratio between yes- and non-events
train_frac = 0.1
sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=42)
subset_idx, _ = next(sss.split(X, y))
X_sub = X.iloc[subset_idx]
y_sub = y.iloc[subset_idx] 

# Train the considered machine learning models
dir_out_temp = os.path.join(git_repo, dir_out, model_2_train)
results = train_with_nested_cv_and_optuna(
      X_sub,
      y_sub,
      model_type=model_2_train,
      dir_out=dir_out_temp,
      n_trials = 20,
      n_outer = 5,
      n_inner = 3,
      n_repeats = 1
      )