import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import metview as mv
import joblib

############################################################################################################
# CODE DESCRIPTION
# 18_prob_ff_hydro_long_fc_compute_poff.py computes the short-range forecasts for the Probability of Flash Floods (PoFF).

# Usage: python3 18_prob_ff_hydro_long_fc_compute_poff.py

# Runtime: ~ 30 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# base_date (date, in YYYYMMDD format): base date to consider for the computation of PoFF.
# step_f_start (integer, in hours): first final step of the accumulation period to consider. 
# step_f_final (integer, in hours): last final step of the accumulation period to consider. 
# delta (float, positive): radius of the area  around the considered grid-box.
# model_name (string): name of the model to train.
# git_repo (string): repository's local path
# file_in_adj_gb (string): relative path of the file containing the adjacent grid-boxes for each grid-box in era5's grid.
# file_in_sdfor (string): relative path of the file containing the standard deviation of the filtered  sub-grid orography.
# dir_in_tp (string): relative path of the directory containing 1s where sum(tp from ERA5-ecPoint) = 0; 0s otherwise.
# dir_in_tp_prob (string): relative path of the directory containing the probability of exceeding a certain return period.
# dir_in_swvl (string): relative path of the directory containing the percentage of soil saturation.
# dir_in_lai (string): relative path of the directory containing the leaf area index.
# dir_in_model (string): relative path of the directory containing the model to consider.
# dir_out (string): relative path containing the point data table for the considered year. 

############################################################################################################
# INPUT PARAMETERS
base_date = datetime(2021,7,16)
step_f_start = 24
step_f_final = 120
model_name = "gradient_boosting_xgboost"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_adj_gb = "data/raw/adjacent_gb/era5_delta_0.5.npy"
file_in_sdfor = "data/raw/reanalysis/era5/orography/sdfor.grib"
dir_in_tp = "data/raw/reanalysis/era5_ecpoint/tp_24h_long_fc"
dir_in_tp_prob = "data/processed/05_tp_prob_exceed_rp_long_fc"
dir_in_swvl = "data/processed/07_swvl_1m_long_fc"
dir_in_lai = "data/raw/reanalysis/era5/lai"
dir_in_model = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
dir_out = "data/processed/18_prob_ff_hydro_long_fc_compute_poff"
############################################################################################################


# Read the mask for the adjacent grid-boxes for each grid-box in the grid
area_global = np.load(f"{git_repo}/{file_in_adj_gb}", allow_pickle=True)

# Reading sdfor
sdfor = mv.values(mv.read(git_repo + "/" + file_in_sdfor))
mask_grib = (mv.read(git_repo + "/" + file_in_sdfor)) * 0

# Computing the PoFF
for step_f in range(step_f_start, step_f_final + 1, 24):

      step_s = step_f - 24

      print(f'\nComputing and saving the PoFF for {base_date.strftime("%Y%m%d")} at 00 UTC (t + {step_s}, t + {step_f})')

      for loss_func in ["bce", "weighted_bce"]:

            for eval_metric in ["auc", "auprc"]:

                  # Load the data-driven model
                  print(f"Loding the data-driven model - {model_name} - for {loss_func} and {eval_metric}")
                  file_in_model = f"{git_repo}/{dir_in_model}/{loss_func}/{eval_metric}/{model_name}" # future improvement: add the possibility to have the neural network
                  model = joblib.load(f"{file_in_model}/model.joblib")

                  # Retrieving the land-sea mask
                  lsm = mv.retrieve(
                        class_ = "ea",
                        date = "1940-01-01",
                        expver = "1",
                        levtype = "sfc",
                        param = 172.128,
                        step = "0",
                        stream = "oper",
                        time = "6",
                        type = "fc"
                        )
                  lsm = mv.bitmap( (lsm>0) , 0)

                  # Reading the rainfall probabilities of exceeding a certain return period
                  file_in_tp_prob_1 = git_repo + "/" + dir_in_tp_prob + "/" + str(1) + "rp/" + base_date.strftime("%Y%m") + "/prob_exceed_rp_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
                  tp_prob_1 = mv.values(mv.read(file_in_tp_prob_1))

                  file_in_tp_prob_50 = git_repo + "/" + dir_in_tp_prob + "/" + str(50) + "rp/" + base_date.strftime("%Y%m") + "/prob_exceed_rp_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
                  tp_prob_50 = mv.values(mv.read(file_in_tp_prob_50))

                  tp_prob_max_1_adj_gb = []
                  tp_prob_max_50_adj_gb = []
                  for i in range(len(tp_prob_1)):
                        area_ind = area_global[i]
                        tp_prob_max_1_adj_gb.append(np.max(tp_prob_1[area_ind]))
                        tp_prob_max_50_adj_gb.append(np.max(tp_prob_50[area_ind]))

                  # Reading swvl
                  file_in_swvl = git_repo + "/" + dir_in_swvl + "/" + base_date.strftime("%Y%m") + "/swvl_1m_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_s:03d}" + ".grib"
                  swvl = mv.values(mv.read(file_in_swvl))

                  # Reading lai
                  vt = base_date + timedelta(hours = step_s)
                  file_in_lai = git_repo + "/" + dir_in_lai  + "/lai_" + vt.strftime("%m%d") + ".grib"
                  lai = mv.values(mv.read(file_in_lai))

                  # Building the dataframe with the values for the predictors
                  X = pd.DataFrame()
                  X["tp_prob_1"] = tp_prob_1
                  X["tp_prob_max_1_adj_gb"] = tp_prob_max_1_adj_gb
                  X["tp_prob_50"] = tp_prob_50
                  X["tp_prob_max_50_adj_gb"] = tp_prob_max_50_adj_gb
                  X["swvl"] = swvl
                  X["sdfor"] = sdfor
                  X["lai"] = lai

                  # Computing the values of PoFF for the considered accumulation period and setting them in a grib file
                  y_prob = model.predict_proba(X)[:, 1]
                  y_prob_grib = mv.set_values(mask_grib, y_prob)

                  # Assign zero PoFF to those grid-boxes with zero rainfall totals
                  file_in_tp = git_repo + "/" + dir_in_tp + "/" + base_date.strftime("%Y%m") + "/Pt_BC_PERC_" + base_date.strftime("%Y%m%d") + "_" + f"{step_f:03d}" + ".grib2"
                  tp = mv.sum(mv.read(file_in_tp))
                  y_prob_grib = (tp > 0) * y_prob_grib * 100 # to plot probabilities in %
                  y_prob_grib = y_prob_grib * lsm # to plot only values on land

                  # Saving the PoFF fields
                  dir_out_temp = f"{git_repo}/{dir_out}/{loss_func}/{eval_metric}/{model_name}"
                  file_out = f"{dir_out_temp}/poff_{base_date.strftime("%Y%m%d")}_{base_date.strftime("%H")}_{step_f:03d}.grib"
                  os.makedirs(dir_out_temp, exist_ok=True)
                  mv.write(file_out, y_prob_grib)