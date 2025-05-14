#!/bin/bash

################################################################################################################
# CODE DESCRIPTION
# retrieve_reanalysis_era5_ecpoint_tp_24h.sh retrieves from disk the 24-hourly rainfall from ERA5-ecPoint.  
# Runtime: the code takes up to 5 hours per year.

# INPUT PARAMETERS DESCRIPTION
# year (date, in YYYY format): year to retrieve.            
# git_repo (string): repository's local path.
# dir_in_global (string): full path containing the tp reanalysis from ERA5-ecPoint to retrieve.
# dir_out (string): relative path containing the retrieved tp reanalysis.

# INPUT PARAMETERS
year=${1}
git_repo="/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in_global="/ec/vol/ecpoint/mofp/reanalysis/ecpoint/SemiOper/ECMWF_ERA5/0001/Rainfall/024/Code2.0.0_Cal1.0.0/Pt_BC_PERC"
dir_out="data/raw/reanalysis/era5_ecpoint/tp_24h"
################################################################################################################


# Setting output directory
dir_out_temp=${git_repo}/${dir_out}
mkdir -p ${dir_out_temp}

# Retrieving the ERA5_ecPoint data
echo "Retrieving ERA5_ecPoint for ${year}"
cp -r ${dir_in_global}/${year}*  ${dir_out_temp}