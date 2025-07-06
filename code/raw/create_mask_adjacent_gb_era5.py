import os
import numpy as np
import metview as mv

#######################################################################
# CODE DESCRIPTION
# create_mask_adjacent_gb_era5.py creates the mask for the adjecent grid-boxes, given 
# a certain radius, to the grid-boxes in the era5 grid. 

# Usage: python3 create_mask_adjacent_gb_era5.py

# Runtime: ~ 2 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# delta (float, positive, in degrees): radius of the area  around the considered grid-box.
# git_repo (string): repository's local path
# file_in (string): relative of the file containg the grid of interest.
# dir_out (string): relative path containing the point data table for the considered year. 

#######################################################################
# INPUT PARAMETERS
delta = 0.5
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in = "data/raw/mask/usa_era5.grib"
dir_out = "data/raw/adjacent_gb"
#######################################################################


# Loading the grid to consider
mask_grib = mv.read(git_repo + "/" + file_in) * 0
mask_lats = mv.latitudes(mask_grib)
mask_lons = mv.longitudes(mask_grib)

# Defining the radius to consider
mask_lats_u = mask_lats + delta
mask_lats_d = mask_lats - delta
mask_lons_l = mask_lons - delta
mask_lons_r = mask_lons + delta

# Defining the grid-boxes within the radius for each grid-box in the grid
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

# Saving the adjacent grid-boxes mask
dir_out_temp = f"{git_repo}/{dir_out}"
os.makedirs(dir_out_temp, exist_ok=True)
np.save(f"{dir_out_temp}/era5_delta_{delta}", area_global)