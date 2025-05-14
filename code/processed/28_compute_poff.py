import os
from datetime import datetime, timedelta
import pandas as pd
import metview as mv
import joblib
from sklearn.metrics import roc_curve, auc, recall_score, f1_score
from sklearn.calibration import calibration_curve

#############################################################################################################
# CODE DESCRIPTION
# 28_compute_poff.py computes the Probability of Flash Floods (PoFF) using the trained models.
# Runtime: the code takes up to 30 minutes.

# INPUT PARAMETERS DESCRIPTION
# year (integer, in YYYY format): year to consider.
# git_repo (string): repository's local path
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in_ff (string): relative path of the directory containing the accumulated gridded flash flood reports.
# file_in_slor (string): relative path of the file containing the slope of sub-grid orography. 
# file_in_sdfor (string): relative path of the file containing the standard deviation of the filtered  sub-grid orography.
# dir_in_tp (string): relative path of the directory containing 1s where sum(tp from ERA5-ecPoint) = 0; 0s otherwise.
# dir_in_tp_prob (string): relative path of the directory containing the probability of exceeding a certain return period.
# dir_in_swvl (string): relative path of the directory containing the percentage of soil saturation.
# dir_in_lai (string): relative path of the directory containing the leaf area index.
# dir_out (string): relative path containing the point data table for the considered year. 

# INPUT PARAMETERS
year = 2024
model_2_read_list = ["random_forest_xgboost", "random_forest_lightgbm", "gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost"] 
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_slor = "data/raw/reanalysis/era5/orography/slor.grib"
file_in_sdfor = "data/raw/reanalysis/era5/orography/sdfor.grib"
dir_in_tp = "data/raw/reanalysis/era5_ecpoint/tp_24h"
dir_in_tp_prob = "data/compute/08_era5_ecpoint_tp_24h_prob_exceed_rp"
dir_in_swvl = "data/compute/10_swvl_1m_perc"
dir_in_lai = "data/raw/reanalysis/era5/lai"
dir_in_model = "data/compute/25_retrain_ml_test"
dir_out = "data/compute/28_poff"
#############################################################################################################


# Reading the trained model
for model_2_read in model_2_read_list:

      # Uploading the ml model
      file_in = git_repo + "/" + dir_in_model + "/" + model_2_read + "/model.joblib"
      model = joblib.load(file_in)

      # Creating a template for a grib file
      template_grib = mv.read(git_repo + "/" + file_in_sdfor) * 0

      # Reading the slor and sdfor values within the considered domain
      sdfor = mv.values(mv.read(git_repo + "/" + file_in_sdfor))

      # Computing the PoFF
      the_date_start_s = datetime(year, 1, 1)
      the_date_start_f = datetime(year, 12, 30)
      the_date_start = the_date_start_s
      while the_date_start <= the_date_start_f:

            the_date_final = the_date_start + timedelta(hours=24)
            print("Creating the PoFF with - " + model_2_read + " - for " + the_date_start.strftime("%Y%m%d"))
            
            # Reading the rainfall probabilities of exceeding a certain return period
            file_in_tp_prob_1 = git_repo + "/" + dir_in_tp_prob + "/" + str(1) + "rp/" + the_date_final.strftime("%Y%m") + "/prob_exceed_rp_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
            tp_prob_1 = mv.values(mv.read(file_in_tp_prob_1))

            file_in_tp_prob_50 = git_repo + "/" + dir_in_tp_prob + "/" + str(50) + "rp/" + the_date_final.strftime("%Y%m") + "/prob_exceed_rp_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
            tp_prob_50 = mv.values(mv.read(file_in_tp_prob_50))

            # Reading swvl (percentage of water content in soil)
            file_in_swvl = git_repo + "/" + dir_in_swvl + "/" + the_date_start.strftime("%Y") + "/swvl_1m_perc_" + the_date_start.strftime("%Y%m%d%H") + ".grib"
            swvl = mv.values(mv.read(file_in_swvl))

            # Reading lai (leaf area index)
            file_in_lai = git_repo + "/" + dir_in_lai  + "/lai_" + the_date_final.strftime("%m%d") + ".grib"
            lai = mv.values(mv.read(file_in_lai))

            # Building the dataframe with the values for the predictors
            X = pd.DataFrame()
            X["tp_prob_1"] = tp_prob_1.astype(float).round(2)
            X["tp_prob_50"] = tp_prob_50.astype(float).round(2)
            X["swvl"] = swvl.astype(float).round(2)
            X["sdfor"] = sdfor.astype(float).round(2)
            X["lai"] = lai.astype(float).round(2)

            # Computing the values of PoFF for the considered accumulation period and setting them in a grib file
            y_prob = model.predict_proba(X)[:, 1]
            y_prob_grib = mv.set_values(template_grib, y_prob)

            # Assign zero PoFF to those grid-boxes with zero rainfall totals
            file_in_tp = git_repo + "/" + dir_in_tp + "/" + the_date_final.strftime("%Y%m") + "/Pt_BC_PERC_" + the_date_final.strftime("%Y%m%d") + "_024.grib2"
            tp = mv.sum(mv.read(file_in_tp))
            y_prob_grib = (tp > 0) * y_prob_grib * 100 # to plot probabilities in %

            # Saving the PoFF fields
            dir_out_temp = git_repo + "/" + dir_out + "/" + model_2_read + "/" + str(year)
            os.makedirs(dir_out_temp, exist_ok=True)
            mv.write(dir_out_temp + "/poff_" + the_date_final.strftime("%Y%m%d") + "_024.grib", y_prob_grib)
            
            the_date_start = the_date_start + timedelta(hours=24)