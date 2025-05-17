#!/bin/bash

#######################################################################
# CODE DESCRIPTION
# retrieve_reanalysis_era5_swvl_short_fc.sh retrieves from MARS the volumetric soil 
# water from ERA5's (31 km resolution) short-range forecasts, for the first 3 soil levels.

# Usage: ./retrieve_reanalysis_era5_swvl_short_fc.sh

# Runtime: ~ 1 hour per year.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year_s (year, in YYYY format): start year to retrieve.
# year_f (year, in YYYY format): final year to retrieve.
# git_repo (string): repository's local path.
# dir_out (string): relative path containing the retrieved forecasts.

#######################################################################
# INPUT PARAMETERS
year_s=2001
year_f=2024
git_repo="/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_out="data/raw/reanalysis/era5/swvl_short_fc"
#######################################################################


# Retrieveing the soil moisture from ERA5-LAND
for year in $(seq ${year_s} ${year_f}); do

      for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
      
            dir_db_temp="${git_repo}/${dir_out}/temp_dir"
            mkdir -p ${dir_db_temp}

mars <<EOF
      retrieve,
            class=ea,
            date=${year}${month}01/to/${year}${month}31,
            expver=1,
            levtype=sfc,
            param=39.128/40.128/41.128,
            stream=oper,
            time=0/to/23/by/1,
            type=an,
            target="${dir_db_temp}/{shortName}_[date]_[time].grib"
EOF

            echo " "
            echo "Creating the database..."
            for file_name in `ls ${dir_db_temp}`; do    
                  
                  param_short_name="$(cut -d'_' -f1 <<<"$file_name")"
                  param_date="$(cut -d'_' -f2 <<<"$file_name")"
                  param_time_ext="$(cut -d'_' -f3 <<<"$file_name")"    
                  param_time="$(cut -d'.' -f1 <<<"$param_time_ext")"  
                  let param_time=${param_time}/100
                  param_time=$(printf "%02d" ${param_time})
                  
                  temp_dir="${git_repo}/${dir_out}/${param_short_name}/${year}/${param_date}"
                  mkdir -p ${temp_dir}
                  mv "${dir_db_temp}/${file_name}" "${temp_dir}/${param_short_name}_${param_date}_${param_time}.grib"

            done

            rm -rf ${dir_db_temp}

      done

done