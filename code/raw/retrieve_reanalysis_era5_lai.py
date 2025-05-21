import os
from datetime import datetime, timedelta
import metview as mv

#######################################################################
# CODE DESCRIPTION
# retrieve_reanalysis_era5_lai.py retrieves from MARS the raw datasets needed to 
# compute the leaf area index. The fields change every day, but they are the same 
# every year.

# Usage: python3 retrieve_reanalysis_era5_lai.py

# Runtime: ~ 1 hour.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# git_repo (string): repository's local path.
# dir_out (string): relative path containing the computed leaf area index.

#######################################################################
# INPUT PARAMETERS
git_repo="/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_out="data/raw/reanalysis/era5/lai"
#######################################################################


# Setting output directory
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
    os.makedirs(dir_out_temp)

# Computing the "overall" lai values (including low and high vegetation)
print()
print("Computing the lai values for:")
date_s = datetime(1940,1,1)
date_f = datetime(1940,12,31)
the_date = date_s
while the_date <= date_f:

      print(" - " + the_date.strftime("%m-%d"))

      # Retrieving from Mars the cover and lai for low and high vegetation
      lai_lv = mv.retrieve(
            class_ = "ea",
            date = the_date.strftime("%Y-%m-%d"),
            expver = 1,
            levtype = "sfc",
            param = "66.128",
            stream = "oper",
            time = "00:00:00",
            type = "an"
            )

      lai_hv = mv.retrieve(
            class_ = "ea",
            date = the_date.strftime("%Y-%m-%d"),
            expver = 1,
            levtype = "sfc",
            param = "67.128",
            stream = "oper",
            time = "00:00:00",
            type = "an"
            )

      cvl = mv.retrieve(
            class_ = "ea",
            date = the_date.strftime("%Y-%m-%d"),
            expver = 1,
            levtype = "sfc",
            param = "27.128",
            stream = "oper",
            time = "00:00:00",
            type = "an"
            )

      cvh = mv.retrieve(
            class_ = "ea",
            date = the_date.strftime("%Y-%m-%d"),
            expver = 1,
            levtype = "sfc",
            param = "28.128",
            stream = "oper",
            time = "00:00:00",
            type = "an"
            )
      
      # Computing the "overall" lai values
      lai = lai_lv * cvl + lai_hv * cvh

      # Saving the "overall" lai values
      file_out = dir_out_temp + "/lai_" + the_date.strftime("%m%d") + ".grib"
      mv.write(file_out, lai)

      the_date = the_date + timedelta(days=1)