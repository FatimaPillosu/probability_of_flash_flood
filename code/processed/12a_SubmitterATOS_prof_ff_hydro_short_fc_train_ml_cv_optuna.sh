#!/bin/bash

#SBATCH --job-name=cv_optuna
#SBATCH --output=log_atos/cv_optuna-%J.out
#SBATCH --error=log_atos/cv_optuna-%J.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

python3 12_prob_ff_hydro_short_fc_train_ml_cv_optuna.py