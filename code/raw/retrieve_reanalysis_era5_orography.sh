#!/bin/bash

#######################################################################
# CODE DESCRIPTION
# retrieve_reanalysis_era5_orography.sh retrieves from MARS orography  
# parameters from ERA5 (31 km resolution). The fields are static.

# Usage: ./retrieve_reanalysis_era5_orography.sh

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# git_repo (string): repository's local path.
# dir_out (string): relative path of the file containing the orography parameters.

#######################################################################
# INPUT PARAMETERS
git_repo="/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_out="data/raw/reanalysis/era5/orography"
#######################################################################


# Setting the output directory
dir_out_temp=${git_repo}/${dir_out}
mkdir -p ${dir_out_temp}

# Retrieving the orography fields from MARS 
mars <<EOF

    retrieve,
        class=ea,
        date=19400101,
        expver=1,
        levtype=sfc,
        param=74.128,
        step=0,
        stream=oper,
        time=0,
        type=an,
        target="${dir_out_temp}/sdfor.grib"

        retrieve,
        class=ea,
        date=19400101,
        expver=1,
        levtype=sfc,
        param=163.128,
        step=0,
        stream=oper,
        time=0,
        type=an,
        target="${dir_out_temp}/slor.grib"

EOF