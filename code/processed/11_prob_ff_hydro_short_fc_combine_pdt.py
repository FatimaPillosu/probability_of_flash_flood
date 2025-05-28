import os
import numpy as np
import pandas as pd
import metview as mv

#######################################################################
# CODE DESCRIPTION
# 11_prob_ff_hydro_short_fc_combine_pdt.py combines the point data tables (for 
# short-range forecasts) for single years to create the full required training dataset.

# Usage: python3 11_prob_ff_hydro_short_fc_combine_pdt.py

# Runtime: ~ 5 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

#  INPUT PARAMETERS DESCRIPTION
# year_s (integer): start year to consider.
# year_f (integer): final year to consider.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the pdt for each year.
# dir_out (string): relative path of the directory containing the combined pdt.

#######################################################################
# INPUT PARAMETERS
year_s = 2021
year_f = 2024
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/10_prob_ff_hydro_short_fc_pdt_year"
dir_out = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt"
#######################################################################


# Defining latitudes, longitudes, and number of grid-points in the considered domain
mask = mv.read(git_repo + "/" + file_in_mask)
ind_mask = np.where(mv.values(mask) == 1)[0]
lats_mask = mv.latitudes(mask)[ind_mask]
lons_mask = (mv.longitudes(mask) - 360)[ind_mask]
num_gp_mask = ind_mask.shape[0]

# Determining the name for the training period 
pdt_all = pd.DataFrame() # initializing the variable that will contained the merged yearly PDTs
print("Merging the yearly PDTs")
for year in range(year_s, year_f+1):

      print(" - Considering year:"+ str(year))

      # Reading the pdt for specific years, select the variables to consider as predictors, and convert the dataframe to a 2-d numpy array
      file_in = git_repo + "/" + dir_in + "/pdt_" + str(year) + ".csv"
      pdt_temp = pd.read_csv(file_in)

      # Converting the counts of point flash flood reports in each grid-box into a binary value (i.e., 0 when there are no reports in the grid-box and 1 when there is at least one report)
      pdt_temp["ff"] = np.where(pdt_temp["ff"] > 0, 1, 0)

      # Eliminate all the points with "tp_greater_0=0" because the rainfall totals was zero and, hence, there should not be flash floods
      pdt_temp = pdt_temp[pdt_temp["tp_greater_0"] > 0]

      # Removing rows containing NaN values in one or more columns
      pdt_temp = pdt_temp.dropna()

      # Merging the yearly pdt
      pdt_all = pd.concat([pdt_all, pdt_temp], ignore_index=True)

print(f"N. of data points in the training period between {year_s} and {year_f}: {len(pdt_all)}")
print(f"N. of flash flood reports: {pdt_all['ff'].sum()}")
print(f"Climatological frequency for flash floods: {( pdt_all['ff'].sum() / len(pdt_all) ) * 100}%")

# Saving the merged PDT for all considered years
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
      os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/pdt_" + str(year_s) + "_" + str(year_f) + ".csv"
pdt_all.to_csv(file_out, index=False)