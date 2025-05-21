#!/bin/bash

#SBATCH --job-name=swvl_1m_long_fc
#SBATCH --output=log_atos/swvl_1m_long_fc-%J.out
#SBATCH --error=log_atos/swvl_1m_long_fc-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
Year=${1}

python3 07_swvl_1m_long_fc.py ${Year}