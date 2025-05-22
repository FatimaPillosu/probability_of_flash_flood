#!/bin/bash

#SBATCH --job-name=prob_ff_meteo_verif_short_fc
#SBATCH --output=log_atos/prob_ff_meteo_verif_short_fc-%J.out
#SBATCH --error=log_atos/prob_ff_meteo_verif_short_fc-%J.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
rp=${1}

python3 08_prob_ff_meteo_verif_short_fc.py ${rp}