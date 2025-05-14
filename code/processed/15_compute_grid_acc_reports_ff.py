import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import metview as mv

####################################################################################
# CODE DESCRIPTION
# 15_compute_grid_acc_reports_ff.py computes the gridded accumulated flash flood reports 
# based on the point accumulated ones.
# Runtime: the code takes up to 1 hour.

# INPUT PARAMETERS DESCRIPTION
# year_s (integer): start year to consider.
# year_f (integer): final year to consider.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the point accumulated flash flood reports.
# dir_out (string): relative path of the directory containing the gridded accumulated flash flood reports.

# INPUT PARAMETERS
year_s = 2001
year_f = 2024
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in = "data/compute/14_point_acc_reports_ff"
dir_out = "data/compute/15_grid_acc_reports_ff"
####################################################################################

# Reading the mask
mask = mv.read("/home/mofp/vol_ecpoint_dev/mofp/papers_2_write/PoFF_USA/data/raw/reanalysis/era5/orography/slor.grib") * 0
mask_vals = mv.values(mask)
lats = mv.latitudes(mask)
lons= mv.longitudes(mask)
lats_unique = np.unique(lats)
lons_unique = np.unique(lons)

# Defining the accumulation periods to consider
the_date_start_s = datetime(year_s,1,1)
the_date_start_f = datetime(year_f,12,31)

# Creating gridded accumulated flash flood reports
print()
print("Computing the gridded accumulated flash flood reports for the 24-hourly period ending:")
area = []
tot_point_ff = 0
tot_grid_ff = 0
the_date_start = the_date_start_s
while the_date_start <= the_date_start_f:

      the_date_final = the_date_start + timedelta(hours=24)
      print(" - on " + the_date_final.strftime("%Y-%m-%d") + " at " + the_date_final.strftime("%H") + " UTC")

      # Initializing the gridded field where to store the flash flood reports
      ff_grid = mv.values((mask == 1) * 0)

      # Reading the point accumulated flash flood reports
      file_in = git_repo + "/" + dir_in + "/" + the_date_final.strftime("%Y") + "/point_acc_reports_ff_" + the_date_final.strftime("%Y%m%d%H") + ".csv"
      if os.path.exists(file_in):
            
            ff = pd.read_csv(file_in)
            len_ff = len(ff)

            for ind in range(len_ff):

                  lat_ff_1 = ff["BEGIN_LAT"].iloc[ind]
                  lat_ff_2 = ff["END_LAT"].iloc[ind]
                  lon_ff_1 = ff["BEGIN_LON"].iloc[ind] + 360
                  lon_ff_2 = ff["END_LON"].iloc[ind] + 360

                  info_1 = mv.nearest_gridpoint_info(mask, lat_ff_1, lon_ff_1)
                  info_2 = mv.nearest_gridpoint_info(mask, lat_ff_2, lon_ff_2)

                  if (info_1[0] is not None) and (info_2[0] is not None):

                        index_ff_grid_1 = int(mv.nearest_gridpoint_info(mask, lat_ff_1, lon_ff_1)[0]["index"])
                        lat_1 = lats[index_ff_grid_1]
                        lon_1 = lons[index_ff_grid_1]

                        index_ff_grid_2 = int(mv.nearest_gridpoint_info(mask, lat_ff_2, lon_ff_2)[0]["index"])
                        lat_2 = lats[index_ff_grid_2]
                        lon_2 = lons[index_ff_grid_2]

                        lats_ind = np.where( (lats_unique <= lat_1) & (lats_unique >= lat_2) )[0]
                        lats_unique_temp = lats_unique[lats_ind]
                        
                        for lat_temp in lats_unique_temp:
                              
                              lons_unique = lons[ np.where(lats == lat_temp)[0] ]
                              lons_ind = np.where( (lons_unique >= lon_1) & (lons_unique <= lon_2) )[0]
                              lons_unique_temp = lons_unique[lons_ind]
                              
                              for lon_temp in lons_unique_temp:

                                    ind_ff = int(mv.nearest_gridpoint_info(mask, lat_temp, lon_temp)[0]["index"])
                                    ff_grid[ind_ff] = ff_grid[ind_ff] + 1

            tot_point_ff = tot_point_ff + len_ff
            tot_grid_ff = tot_grid_ff + np.nansum(ff_grid)

            # Converting the gridded field with flash flood reports into grib
            ff_grid = mv.set_values(mask, ff_grid)

            # Saving grib with gridded accumulated flash flood reports
            dir_out_temp = git_repo + "/" + dir_out + "/" + the_date_final.strftime("%Y")
            if not os.path.exists(dir_out_temp):
                  os.makedirs(dir_out_temp)
            file_out = dir_out_temp + "/grid_acc_reports_ff_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
            mv.write(file_out, ff_grid)

      else:
            
            print("     Note: accumulation period with no flash flood reports.")

      the_date_start = the_date_start + timedelta(hours=24)

print(f"Total n. of point reports (in csv, all USA): {tot_point_ff}")
print(f"Total n. of point reports (expanded reported area, all USA): {tot_grid_ff}")