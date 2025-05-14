#!/bin/bash

#SBATCH --job-name=compute_swvl_1m_perc
#SBATCH --output=LogATOS/compute_swvl_1m_perc-%J.out
#SBATCH --error=LogATOS/compute_swvl_1m_perc-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
Year=${1}

python3 10_compute_swvl_1m_perc.py ${Year}