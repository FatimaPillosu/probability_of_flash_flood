#!/bin/bash

#SBATCH --job-name=swvl_1m_perc_fc
#SBATCH --output=LogATOS/swvl_1m_perc_fc-%J.out
#SBATCH --error=LogATOS/swvl_1m_perc_fc-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
Year=${1}

python3 41_compute_swvl_1m_perc_fc.py ${Year}