import os
import sys
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.verif_scores import (contingency_table_probabilistic, 
                                                      precision,
                                                      hit_rate, 
                                                      false_alarm_rate,
                                                      reliability_diagram,
                                                      aroc_trapezium
                                                      )
from matplotlib.ticker import FuncFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

########################################################################################
# CODE DESCRIPTION
# 25_prob_ff_hydro_long_fc_verif.py plots the verification results for long-range forecasts, over the verification dataset.
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall score)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
#     - precision-recall curve (breakdown score for imbalanced datasets)
#     - area under the precision-recall curve (overall performance)

# Usage: python3 25_prob_ff_hydro_long_fc_verif.py

# Runtime: ~ 5 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# verif_period (string): verification period, in the format of "yearS_yearF".
# step_f_start (integer, in hours): first final step of the accumulation period to consider. 
# step_f_final (integer, in hours): last final step of the accumulation period to consider. 
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

# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

########################################################################################
# INPUT PARAMETERS
verif_period = "2021_2024"
step_f_start = 24
step_f_final = 120
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
model_name = "gradient_boosting_xgboost"
loss_func_list = ["bce", "weighted_bce"]
eval_metric_list = ["auc", "auprc"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_model = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
dir_in_pdt = "data/processed/16_prob_ff_hydro_long_fc_combine_pdt"
dir_out = "data/plot/25_prob_ff_hydro_long_fc_verif"
##############################################################################################################


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


# Creating the verification plots
for loss_func in ["bce", "weighted_bce"]:

      for eval_metric in ["auc", "auprc"]:

            print(f"\nCreating verification plots for loss_fun = {loss_func}, and eval_metric = {eval_metric}")

            # Load the data-driven model
            file_in_model = f"{git_repo}/{dir_in_model}/{loss_func}/{eval_metric}/{model_name}" # future improvement: add the possibility to have the neural network
            model = joblib.load(f"{file_in_model}/model.joblib")
            prob_thr = np.load(f"{file_in_model}/best_thr.npy") * 100

            # Creating the input/output directory
            dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}'
            os.makedirs(dir_out_temp, exist_ok=True)

            # Initialising the variables storing the overall scores for all lead times
            auprc_test_all = []
            aroc_test_all = []
            fb_test_all = []

            for step_f in range(step_f_start, step_f_final + 1, 24):

                  # Read the point data table
                  logger.info(f"\nReading the pdt for t+{step_f}")
                  file_in_test = f"{git_repo}/{dir_in_pdt}/pdt_{verif_period}_{step_f:03d}.csv"
                  X, obs_test = load_data(file_in_test, feature_cols, target_col)
                  obs_test = obs_test.to_numpy()

                  # Create the forecasts
                  fc_prob_test = model.predict_proba(X)[:, 1] * 100
                  fc_test = fc_prob_test > prob_thr

                  # Computing the contingency table
                  h_test, fa_test, m_test, cn_test = contingency_table_probabilistic(obs_test, fc_prob_test, 100)
                  print()
                  
                  # Plotting the precision-recall curve
                  plt.figure(figsize=(6.5, 6))
                  p_test = precision(h_test, fa_test)
                  hr_test = hit_rate(h_test, m_test)
                  ref_test = np.sum(obs_test) / len(obs_test)
                  auprc_test_all.append(average_precision_score(obs_test, fc_prob_test))
                  plt.plot(hr_test, p_test, "--o", color = "#00B0F0", lw = 2, ms=4)
                  plt.plot([0,1], [ref_test, ref_test], "--", color = "#333333", lw = 2)
                  plt.xlabel("Recall", color = "#333333", fontsize = 28)
                  plt.ylabel("Precision", color = "#333333", fontsize = 28)
                  plt.tick_params(axis='x', colors='#333333', labelsize=28)
                  plt.tick_params(axis='y', colors='#333333', labelsize=28)
                  plt.xticks(np.arange(0, 1.01, 0.2))
                  plt.yticks(np.arange(0, 1.01, 0.2))
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-0.02,1.02])
                  plt.ylim([-0.02,1.02])
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/pr_curve_{step_f}.png', dpi=1000)
                  plt.close()


                  # Plotting the ROC curve - Trapezium and Continuous
                  plt.figure(figsize=(6.5, 6))
                  hr_test = hit_rate(h_test, m_test)
                  far_test = false_alarm_rate(fa_test, cn_test)
                  aroc_test = aroc_trapezium(hr_test, far_test)
                  plt.plot(far_test, hr_test, "-o", color = "#00B0F0", lw = 2, ms=4, label = f"{aroc_test:.3f}")
                  far_test_c, hr_test_c, thr_roc = roc_curve(obs_test, fc_prob_test)
                  aroc_test_c = auc(far_test_c, hr_test_c)
                  aroc_test_all.append(aroc_test_c)
                  plt.plot(far_test_c, hr_test_c, "--", color = "#00B0F0", lw = 2, ms=2, label = f"{aroc_test_c:.3f}")
                  plt.plot([0,1], [0, 1], "-", color = "#333333", lw = 1)
                  plt.xlabel("False Alarm Rate", color = "#333333", fontsize = 28)
                  plt.ylabel("Hit Rate", color = "#333333", fontsize = 28)
                  plt.xticks(np.arange(0, 1.01, 0.2))
                  plt.yticks(np.arange(0, 1.01, 0.2))
                  plt.tick_params(axis='x', colors='#333333', labelsize=28)
                  plt.tick_params(axis='y', colors='#333333', labelsize=28)
                  fmt = FuncFormatter(lambda val, pos: f"{val:.1f}".lstrip("0") if abs(val) < 1 else f"{val:.1f}")
                  ax = plt.gca()
                  ax.xaxis.set_major_formatter(fmt)
                  ax.yaxis.set_major_formatter(fmt)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-0.02,1.02])
                  plt.ylim([-0.02,1.02])
                  plt.legend(title = "AROC", title_fontsize=24, fontsize=24, frameon=False, loc='lower right')
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/roc_curve_{step_f}.png', dpi=1000)
                  plt.close()


                  # Plotting the reliability diagram
                  fig, ax = plt.subplots(figsize=(6.5, 6))
                  mean_prob_fc_test, mean_freq_obs_test, sharpness_test = reliability_diagram(obs_test, fc_prob_test)
                  plt.plot(mean_prob_fc_test, mean_freq_obs_test * 100, "-o", color = "#00B0F0", lw = 2, ms=4)
                  plt.plot([0,100], [0, 100], color = "#333333", lw = 1)
                  plt.xlabel("Forecast probability", color = "#333333", fontsize = 28)
                  plt.ylabel("Observation frequency", color = "#333333", fontsize = 28)
                  plt.tick_params(axis='x', colors='#333333', labelsize=28)
                  plt.tick_params(axis='y', colors='#333333', labelsize=28)
                  ticks = np.arange(0, 101, 5)
                  labels = [str(t) if t % 20 == 0 else '' for t in ticks]
                  plt.xticks(ticks, labels)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-1,101])
                  plt.ylim([-1,101])
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/reliability_diagram_{step_f}.png', dpi=1000)
                  plt.close()

                  # Computing the frequency bias
                  fb_test_all.append( np.sum(fc_test) / np.sum(obs_test))


            # Plotting the overall scores - AROC
            plt.plot(np.arange(step_f_start, step_f_final + 1, 24), aroc_test_all, "-o", color = "#00B0F0", lw = 2, ms=4)
            plt.ylabel("AROC", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(np.arange(step_f_start, step_f_final + 1, 24))
            plt.grid(axis='y', linewidth=0.5, color='gainsboro')
            plt.ylim([0.5,1])
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/aroc.png', dpi=1000)
            plt.close()
      
            # Plotting the overall scores - AUPRC
            plt.plot(np.arange(step_f_start, step_f_final + 1, 24), auprc_test_all, "-o", color = "#00B0F0", lw = 2, ms=4)
            plt.ylabel("AUPRC", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(np.arange(step_f_start, step_f_final + 1, 24))
            plt.grid(axis='y', linewidth=0.5, color='gainsboro')
            plt.ylim([-0.005,0.07])
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/auprc.png', dpi=1000)
            plt.close()

            # Plotting the overall scores - FB
            plt.plot(np.arange(step_f_start, step_f_final + 1, 24), fb_test_all, "-o", color = "#00B0F0", lw = 2, ms=4)
            plt.ylabel("FB", color = "#333333", fontsize = 12)
            plt.tick_params(axis='x', colors='#333333', labelsize=12)
            plt.tick_params(axis='y', colors='#333333', labelsize=12)
            plt.xticks(np.arange(step_f_start, step_f_final + 1, 24))
            plt.grid(axis='y', linewidth=0.5, color='gainsboro')
            plt.tight_layout()
            plt.savefig(f'{dir_out_temp}/fb.png', dpi=1000)
            plt.close()