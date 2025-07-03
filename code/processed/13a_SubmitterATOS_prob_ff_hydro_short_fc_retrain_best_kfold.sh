#!/bin/bash

#SBATCH --job-name=retrain_ml
#SBATCH --output=log_atos/retrain_ml-%J.out
#SBATCH --error=log_atos/retrain_ml-%J.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=ng
#SBATCH --gpus=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatima.pillosu@ecmwf.int

# INPUTS
model_2_train=${1}
loss_fn_choice=${2}
eval_metric=${3}
rep_to_run=${4}
outer_fold_to_run=${5}

python3 13_prob_ff_hydro_short_fc_retrain_best_kfold.py ${model_2_train} ${loss_fn_choice} ${eval_metric} ${rep_to_run} ${outer_fold_to_run}