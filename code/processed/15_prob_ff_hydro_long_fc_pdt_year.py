import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import metview as mv

################################################################################################
# CODE DESCRIPTION
# 15_prob_ff_hydro_long_fc_pdt_year.py computes the training dataset (Point Data Table) for a specific year to 
# compute the long-range data-driven forecasts of areas at risk of flash floods.

# Usage: python3 15_prob_ff_hydro_long_fc_pdt_year.py 2024

# Runtime: ~ 30 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year (integer, in YYYY format): year to consider.
# step_f (integer, in hours): final step of the accumulation period to consider. 
# delta (float, positive): radius of the area  around the considered grid-box.
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

################################################################################################
# INPUT PARAMETERS
year = int(sys.argv[1])
step_f = int(sys.argv[2])
delta = 0.5
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in_ff = "data/processed/03_grid_acc_reports_ff"
file_in_slor = "data/raw/reanalysis/era5/orography/slor.grib"
file_in_sdfor = "data/raw/reanalysis/era5/orography/sdfor.grib"
dir_in_tp = "data/raw/reanalysis/era5_ecpoint/tp_24h_long_fc"
dir_in_tp_prob = "data/processed/05_tp_prob_exceed_rp_long_fc"
dir_in_swvl = "data/processed/07_swvl_1m_long_fc"
dir_in_lai = "data/raw/reanalysis/era5/lai"
dir_out = "data/processed/15_prob_ff_hydro_long_fc_pdt_year"
################################################################################################


print()
print("Computing the training dataset (Point Data Table) for: " + str(year))

# Start step of the accumulation period
step_s = step_f - 24

# Determining the grid-boxes within the area around each grid-box in the domain
mask_grib = mv.read(git_repo + "/" + file_in_mask)
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

# Defining the domain's mask, with lats/lons, and the areas for each grid-box in the domain
ind_mask = np.where(mv.values(mask_grib) ==1)[0]
lats = mask_lats[ind_mask]
lons = mask_lons[ind_mask] 
area = area_global[ind_mask] 

# Reading the sdfor values within the considered domain
print(" - Reading the sdfor values within the considered domain")
sdfor = mv.values(mv.read(git_repo + "/" + file_in_sdfor))[ind_mask]

# Initializing the variables that contain the predictors that change over different accumulation periods
base_date_all = np.array([]) 
base_time_all = np.array([]) 
step_f_all = np.array([]) 
lats_all = np.array([]) 
lons_all = np.array([]) 
ff_all = np.array([])
tp_greater_0_all  = np.array([])
tp_prob_1_all = np.array([])
tp_prob_50_all = np.array([])
tp_prob_max_1_adj_gb_all = np.array([])
tp_prob_max_50_adj_gb_all = np.array([])
swvl_all = np.array([])
sdfor_all = np.array([])
lai_all = np.array([])

# Creating the point data table
base_date_s = datetime(year, 1, 1, 0)
base_date_f = datetime(year, 12, 31, 0)
base_date = base_date_s
while base_date <= base_date_f:

      print(" - Reading the predictand (flash flood reports) and the predictors (e.g. rainfall probabilities, swvl, slor, sdfor, and lai) for " + base_date.strftime("%Y%m%d"))
      
      # Reading the flash flood reports
      vt = base_date + timedelta(hours = step_f)
      file_in_ff = git_repo + "/" + dir_in_ff + "/" + vt.strftime("%Y") + "/grid_acc_reports_ff_" + vt.strftime("%Y%m%d") + "_" + vt.strftime("%H")  + ".grib"
      
      if os.path.exists(file_in_ff): # building the point data table only for those days with some flash flood reports in the domain
            
            ff = mv.values(mv.read(file_in_ff))[ind_mask]
            
            # Reading the rainfall totals from ERA5-ecPoint and selecting the grid-boxes with zero rainfall totals
            file_in_tp = git_repo + "/" + dir_in_tp + "/" + base_date.strftime("%Y%m") + "/Pt_BC_PERC_" + base_date.strftime("%Y%m%d") + "_" + f"{step_f:03d}" + ".grib2"
            tp_greater_0 = mv.values((mv.sum(mv.read(file_in_tp)) > 0))[ind_mask]
            
            # Reading the rainfall probabilities of exceeding a certain return period
            file_in_tp_prob_1 = git_repo + "/" + dir_in_tp_prob + "/" + str(1) + "rp/" + base_date.strftime("%Y%m") + "/prob_exceed_rp_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
            tp_prob_1_temp = mv.values(mv.read(file_in_tp_prob_1))
            tp_prob_1 = tp_prob_1_temp[ind_mask]
            
            file_in_tp_prob_50 = git_repo + "/" + dir_in_tp_prob + "/" + str(50) + "rp/" + base_date.strftime("%Y%m") + "/prob_exceed_rp_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib"
            tp_prob_50_temp = mv.values(mv.read(file_in_tp_prob_50))
            tp_prob_50 = tp_prob_50_temp[ind_mask]

            tp_prob_max_1_adj_gb = []
            tp_prob_max_50_adj_gb = []
            for i in range(len(ind_mask)):
                  area_ind = area[i]
                  tp_prob_max_1_adj_gb.append(np.max(tp_prob_1_temp[area_ind]))
                  tp_prob_max_50_adj_gb.append(np.max(tp_prob_50_temp[area_ind]))
            tp_prob_max_1_adj_gb = np.array(tp_prob_max_1_adj_gb)
            tp_prob_max_50_adj_gb = np.array(tp_prob_max_50_adj_gb)
            
            # Reading swvl (percentage of water content in soil)
            file_in_swvl = git_repo + "/" + dir_in_swvl + "/" + base_date.strftime("%Y%m") + "/swvl_1m_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_s:03d}" + ".grib"
            swvl = mv.values(mv.read(file_in_swvl))[ind_mask]

            # Reading lai (leaf area index)
            vt_lai = base_date + timedelta(hours = step_s)
            file_in_lai = git_repo + "/" + dir_in_lai  + "/lai_" + vt_lai.strftime("%m%d") + ".grib"
            lai = mv.values(mv.read(file_in_lai))[ind_mask]

            # Concatenating the predicant and predictors values for each day
            base_date_all = np.concatenate((base_date_all, np.array([base_date.strftime("%Y%m%d")] * lats.shape[0])))
            base_time_all = np.concatenate((base_time_all, np.array([base_date.strftime("%H")] * lats.shape[0])))
            step_f_all = np.concatenate((step_f_all, np.array([step_f] * lats.shape[0])))
            lats_all = np.concatenate((lats_all, lats))
            lons_all = np.concatenate((lons_all, lons))
            ff_all = np.concatenate((ff_all, ff))
            tp_greater_0_all = np.concatenate((tp_greater_0_all, tp_greater_0))
            tp_prob_1_all = np.concatenate((tp_prob_1_all, tp_prob_1))
            tp_prob_50_all = np.concatenate((tp_prob_50_all, tp_prob_50))
            tp_prob_max_1_adj_gb_all = np.concatenate((tp_prob_max_1_adj_gb_all, tp_prob_max_1_adj_gb))
            tp_prob_max_50_adj_gb_all = np.concatenate((tp_prob_max_50_adj_gb_all, tp_prob_max_50_adj_gb))
            swvl_all = np.concatenate((swvl_all, swvl))
            sdfor_all = np.concatenate((sdfor_all, sdfor))
            lai_all = np.concatenate((lai_all, lai))

      base_date = base_date + timedelta(hours=24)

# Build the point data table as a pandas dataframe 
pdt = pd.DataFrame()
pdt["base_date"] = base_date_all
pdt["base_time"] = base_time_all
pdt["step_f"] = step_f_all.astype(int)
pdt["lat"] = lats_all.astype(float).round(3)
pdt["lon"] = lons_all.astype(float).round(3)
pdt["ff"] = ff_all.astype(int)
pdt["tp_greater_0"] = tp_greater_0_all.astype(int)
pdt["tp_prob_1"] = tp_prob_1_all.astype(float).round(2)
pdt["tp_prob_max_1_adj_gb"] = tp_prob_max_1_adj_gb_all.astype(float).round(2)
pdt["tp_prob_50"] = tp_prob_50_all.astype(float).round(2)
pdt["tp_prob_max_50_adj_gb"] = tp_prob_max_50_adj_gb_all.astype(float).round(2)
pdt["swvl"] = swvl_all.astype(float).round(2)
pdt["sdfor"] = sdfor_all.astype(float).round(2)
pdt["lai"] = lai_all.astype(float).round(2)

# Saving the point data table as a csv
print("Saving the point data table as a csv")
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
      os.makedirs(dir_out_temp)
pdt.to_csv(dir_out_temp + "/pdt_" + str(year) + "_" + f"{step_f:03d}" + ".csv", index=False)