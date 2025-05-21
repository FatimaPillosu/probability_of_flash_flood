#!/bin/bash

#SBATCH --job-name=prob_ff_meteo_verif_long_fc
#SBATCH --output=log_atos/prob_ff_meteo_verif_long_fc-%J.out
#SBATCH --error=log_atos/prob_ff_meteo_verif_long_fc-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
rp=${1}

python3 11_prob_ff_meteo_verif_long_fc.py ${rp}