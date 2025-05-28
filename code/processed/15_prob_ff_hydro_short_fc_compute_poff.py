import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import metview as mv
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.map_plots import plot_poff

################################################################################################
# CODE DESCRIPTION
# 15_prob_ff_hydro_short_fc_compute_poff.py computes the short-range forecasts for the Probability of Flash Floods 
# (PoFF) using the best trained model.

# Usage: python3 15_prob_ff_hydro_short_fc_compute_poff.py

# Runtime: ~ 30 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# the_date_start (date, in YYYYMMDD format): date to consider for the computation of PoFF.
# delta (float, positive): radius of the area  around the considered grid-box.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path
# file_in_sdfor (string): relative path of the file containing the standard deviation of the filtered  sub-grid orography.
# dir_in_tp (string): relative path of the directory containing 1s where sum(tp from ERA5-ecPoint) = 0; 0s otherwise.
# dir_in_tp_prob (string): relative path of the directory containing the probability of exceeding a certain return period.
# dir_in_swvl (string): relative path of the directory containing the percentage of soil saturation.
# dir_in_lai (string): relative path of the directory containing the leaf area index.
# dir_in_model (string): relative path of the directory containing the best model to compute PoFF.
# dir_out (string): relative path containing the point data table for the considered year. 

################################################################################################
# INPUT PARAMETERS
the_date_start = datetime(2024,10,29)
delta = 0.5
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_sdfor = "data/raw/reanalysis/era5/orography/sdfor.grib"
dir_in_tp = "data/raw/reanalysis/era5_ecpoint/tp_24h_short_fc"
dir_in_tp_prob = "data/processed/04_tp_prob_exceed_rp_short_fc"
dir_in_swvl = "data/processed/06_swvl_1m_short_fc"
dir_in_lai = "data/raw/reanalysis/era5/lai"
dir_in_model = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold_new/gradient_boosting_xgboost"
dir_out = "data/processed/15_prob_ff_hydro_short_fc_compute_poff"
################################################################################################

# Define the output file
the_date_final = the_date_start + timedelta(hours=24)
dir_out_temp = git_repo + "/" + dir_out + "/" + str(the_date_final.strftime("%Y"))
file_out = dir_out_temp + "/poff_" + the_date_final.strftime("%Y%m%d") + "_00.grib"

# Reading or computing the PoFF
if os.path.exists(file_out): # reading
      
      print(f'\nReading the PoFF for {the_date_start.strftime("%Y%m%d")}')
      y_prob_grib = mv.read(file_out)

else: # computing

      print(f'\nComputing and saving the PoFF for {the_date_start.strftime("%Y%m%d")}')
    
      # Uploading the ml model
      file_in = git_repo + "/" + dir_in_model + "/model.joblib"
      model = joblib.load(file_in)

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

      # Defining the grid-boxes within the area around each grid-box in the domain
      mask_grib = mv.read(git_repo + "/" + file_in_sdfor) * 0
      mask_lats = mv.latitudes(mask_grib)
      mask_lons = mv.longitudes(mask_grib)

      mask_lats_u = mask_lats + delta
      mask_lats_d = mask_lats - delta
      mask_lons_l = mask_lons - delta
      mask_lons_r = mask_lons + delta

      area_global = []
      for ind in range(len(mask_lats)):

            print(f"{ind}/{len(mask_lats)}")

            lat = mask_lats[ind]
            lon = mask_lons[ind]
            lat_u = mask_lats_u[ind]
            lat_d = mask_lats_d[ind]
            lon_l = mask_lons_l[ind]
            lon_r = mask_lons_r[ind]

            index_lats = np.where((mask_lats <= lat_u) & (mask_lats >= lat_d))[0]
            mask_lats_temp = mask_lats[index_lats]
            mask_lons_temp = mask_lons[index_lats]

            if lon_l < 0:
                  index_area = np.where((mask_lons_temp >= lon_l+360) | (mask_lons_temp <= lon_r))[0]
            elif lon_r > 360:
                  index_area = np.where((mask_lons_temp >= lon_l) | (mask_lons_temp <= lon_r-360))[0]
            else:
                  index_area = np.where((mask_lons_temp >= lon_l) & (mask_lons_temp <= lon_r))[0]

            area_global.append(index_area)

      area_global = np.array(area_global, dtype=object)

      # Reading sdfor
      sdfor = mv.values(mv.read(git_repo + "/" + file_in_sdfor))

      # Reading the rainfall probabilities of exceeding a certain return period
      file_in_tp_prob_1 = git_repo + "/" + dir_in_tp_prob + "/" + str(1) + "rp/" + the_date_final.strftime("%Y%m") + "/prob_exceed_rp_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
      tp_prob_1 = mv.values(mv.read(file_in_tp_prob_1))

      file_in_tp_prob_50 = git_repo + "/" + dir_in_tp_prob + "/" + str(50) + "rp/" + the_date_final.strftime("%Y%m") + "/prob_exceed_rp_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
      tp_prob_50 = mv.values(mv.read(file_in_tp_prob_50))

      tp_prob_max_1_adj_gb = []
      tp_prob_max_50_adj_gb = []
      for i in range(len(tp_prob_1)):
            area_ind = area_global[i]
            tp_prob_max_1_adj_gb.append(np.max(tp_prob_1[area_ind]))
            tp_prob_max_50_adj_gb.append(np.max(tp_prob_50[area_ind]))

      # Reading swvl
      file_in_swvl = git_repo + "/" + dir_in_swvl + "/" + the_date_start.strftime("%Y") + "/swvl_1m_perc_" + the_date_start.strftime("%Y%m%d%H") + ".grib"
      swvl = mv.values(mv.read(file_in_swvl))

      # Reading lai
      file_in_lai = git_repo + "/" + dir_in_lai  + "/lai_" + the_date_final.strftime("%m%d") + ".grib"
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
      file_in_tp = git_repo + "/" + dir_in_tp + "/" + the_date_start.strftime("%Y%m") + "/Pt_BC_PERC_" + the_date_start.strftime("%Y%m%d") + "_024.grib2"
      tp = mv.sum(mv.read(file_in_tp))
      y_prob_grib = (tp > 0) * y_prob_grib * 100 # to plot probabilities in %
      y_prob_grib = y_prob_grib * lsm # to plot only values on land

      # Saving the PoFF fields
      os.makedirs(dir_out_temp, exist_ok=True)
      mv.write(file_out, y_prob_grib)

# Plotting the PoFF fields
title = f'Probability of Flash Flood for {the_date_start.strftime("%Y%m%d")}'
plot_poff(y_prob_grib, title)