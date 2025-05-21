import os
import shutil

############################################################################################
# CODE DESCRIPTION
# retrieve_tp_24h_climate_1991_2020.py retrieves the 24-hourly rainfall climate computed between 1991 and 
# 2020, from ERA5-ecPoint.  

# Usage: python3 retrieve_tp_24h_climate_1991_2020.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# git_repo (string): repository's local path.
# file_in (string): full path of the file containing the rainfall climatology to retrieve.
# dir_out (string): relative path of the directory containing the rainfall climatology and the computed percentiles.

############################################################################################
# INPUT PARAMETERS
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_global ="/ec/vol/ecpoint/mofp/climate_reference/data/era5_ecpoint/tp_24h_1991_2020"
dir_out = "data/raw/reanalysis/era5_ecpoint/tp_24h_climate_1991_2020"
############################################################################################


# Setting output directory
main_dir_out = git_repo + "/" + dir_out
if not os.path.exists(main_dir_out):
    os.makedirs(main_dir_out)

# Copying rainfall climatology and correspondent return periods
file_in_climate = dir_in_global + "/Climate_RP.grib"
file_out_climate = main_dir_out + "/climate_rp.grib"
shutil.copyfile(file_in_climate, file_out_climate)

file_in_rp = dir_in_global + "/RP.npy"
file_out_rp = main_dir_out + "/rp.npy"
shutil.copyfile(file_in_rp, file_out_rp)