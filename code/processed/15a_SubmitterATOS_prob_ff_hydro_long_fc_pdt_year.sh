#!/bin/bash

#SBATCH --job-name=pdt_year_long_fc
#SBATCH --output=log_atos/pdt_year_long_fc-%J.out
#SBATCH --error=log_atos/pdt_year_long_fc-%J.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
year=${1}
step_f=${2}

python3 15_prob_ff_hydro_long_fc_pdt_year.py ${year} ${step_f}