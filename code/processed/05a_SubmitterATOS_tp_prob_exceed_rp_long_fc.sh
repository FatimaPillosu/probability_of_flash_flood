#!/bin/bash

#SBATCH --job-name=tp_prob_exceed_rp_long_fc
#SBATCH --output=log_atos/tp_prob_exceed_rp_long_fc-%J.out
#SBATCH --error=log_atos/tp_prob_exceed_rp_long_fc-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
Year=${1}

python3 05_tp_prob_exceed_rp_long_fc.py ${Year}