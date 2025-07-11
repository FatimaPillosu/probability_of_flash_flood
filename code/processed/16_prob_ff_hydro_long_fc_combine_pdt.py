import os
import numpy as np
import pandas as pd
import metview as mv

#######################################################################
# CODE DESCRIPTION
# 16_prob_ff_hydro_long_fc_combine_pdt.py combines the point data tables (for 
# long-range forecasts) for single years to create the full required training dataset.

# Usage: python3 16_prob_ff_hydro_long_fc_combine_pdt.py

# Runtime: ~ 5 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

#  INPUT PARAMETERS DESCRIPTION
# year_s (integer): start year to consider.
# year_f (integer): final year to consider.
# step_f_start (integer, in hours): first final step of the accumulation period to consider. 
# step_f_final (integer, in hours): last final step of the accumulation period to consider. 
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the pdt for each year.
# dir_out (string): relative path of the directory containing the combined pdt.

#######################################################################
# INPUT PARAMETERS
year_s = 2021
year_f = 2024
step_f_start = 24
step_f_final = 120
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/15_prob_ff_hydro_long_fc_pdt_year"
dir_out = "data/processed/16_prob_ff_hydro_long_fc_combine_pdt"
#######################################################################


# Defining latitudes, longitudes, and number of grid-points in the considered domain
mask = mv.read(git_repo + "/" + file_in_mask)
ind_mask = np.where(mv.values(mask) == 1)[0]
lats_mask = mv.latitudes(mask)[ind_mask]
lons_mask = (mv.longitudes(mask) - 360)[ind_mask]
num_gp_mask = ind_mask.shape[0]

# Merging the pdt for the considered lead time
for step_f in range(step_f_start, step_f_final + 1, 24):

      # Merging the years in the considered period
      pdt_all = pd.DataFrame() # initializing the variable that will contained the merged yearly PDTs
      print(f"\nMerging the yearly PDTs:")
      for year in range(year_s, year_f+1):

            print(f" -  for {year} and t+{step_f}")

            # Reading the pdt for specific years, select the variables to consider as predictors, and convert the dataframe to a 2-d numpy array
            file_in = git_repo + "/" + dir_in + "/pdt_" + str(year) + "_" + f"{step_f:03d}" + ".csv"
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
      file_out = dir_out_temp + "/pdt_" + str(year_s) + "_" + str(year_f) + "_" + f"{step_f:03d}" + ".csv"
      pdt_all.to_csv(file_out, index=False)