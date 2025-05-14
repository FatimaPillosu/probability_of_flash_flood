#!/bin/bash

#SBATCH --job-name=pdt_fc_year
#SBATCH --output=LogATOS/pdt_fc_year-%J.out
#SBATCH --error=LogATOS/pdt_fc_year-%J.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=nf
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
year=${1}

python3 42_compute_pdt_fc_year.py ${year}