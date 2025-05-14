#!/bin/bash

#SBATCH --job-name=tp_prob_exceed_rp
#SBATCH --output=LogATOS/tp_prob_exceed_rp-%J.out
#SBATCH --error=LogATOS/tp_prob_exceed_rp-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
Year=${1}

python3 08_compute_tp_prob_exceed_rp.py ${Year}