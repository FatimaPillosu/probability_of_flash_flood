#!/bin/bash

#######################################################
# CODE DESCRIPTION
# retrieve_reanalysis_era5_swvl_fc.sh retrieves from MARS the 
# volumetric soil water from ERA5 (31 km resolution) medium-range
# forecasts for the first 3 soil levels.
# Runtime: the code takes up to 1 hour per year.

# INPUT PARAMETERS DESCRIPTION
# year_s (year, in YYYY format): start year to retrieve.
# year_f (year, in YYYY format): final year to retrieve.
# git_repo (string): repository's local path.
# dir_out (string): relative path containing the retrieved forecasts.

# INPUT PARAMETERS
year_s=2021
year_f=2024
git_repo="/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_out="data/raw/reanalysis/era5/swvl_fc"
#######################################################


# Retrieveing the soil moisture from ERA5-LAND
for year in $(seq ${year_s} ${year_f}); do

      for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
      
            dir_db_temp="${git_repo}/${dir_out}/temp_dir"
            mkdir -p ${dir_db_temp}

mars <<EOF
      retrieve,
            class=ea,
            date=${year}${month}01/to/${year}${month}31,
            expver=11,
            levtype=sfc,
            param=39.128/40.128/41.128,
            step=0/24/48/72/96/120,
            stream=oper,
            time=0,
            type=fc,
            target="${dir_db_temp}/{shortName}_[date]_[time]_[step].grib"
EOF

            echo " "
            echo "Creating the database..."
            for file_name in `ls ${dir_db_temp}`; do    
                  
                  param_short_name="$(cut -d'_' -f1 <<<"$file_name")"
                  param_date="$(cut -d'_' -f2 <<<"$file_name")"
                  param_time="$(cut -d'_' -f3 <<<"$file_name")"    
                  let param_time=${param_time}/100
                  param_time=$(printf "%02d" ${param_time})
                  param_step_ext="$(cut -d'_' -f4 <<<"$file_name")"
                  param_step="$(cut -d'.' -f1 <<<"$param_step_ext")" 
                  param_step=$(printf "%03d" ${param_step})
                  
                  temp_dir="${git_repo}/${dir_out}/${param_short_name}/${year}/${param_date}${param_time}"
                  mkdir -p ${temp_dir}
                  mv "${dir_db_temp}/${file_name}" "${temp_dir}/${param_short_name}_${param_date}_${param_time}_${param_step}.grib"

            done

            rm -rf ${dir_db_temp}

      done

done