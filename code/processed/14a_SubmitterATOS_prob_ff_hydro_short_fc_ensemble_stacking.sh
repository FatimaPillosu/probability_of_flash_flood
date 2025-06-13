#!/bin/bash

#SBATCH --job-name=ensemble_stacking
#SBATCH --output=log_atos/retrain_ml-%J.out
#SBATCH --error=log_atos/retrain_ml-%J.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=ng
#SBATCH --gpus=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

python3 14_prob_ff_hydro_short_fc_ensemble_stacking.py