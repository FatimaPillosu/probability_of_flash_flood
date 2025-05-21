#!/bin/bash

rp_list=(1 5 10 20 50 100)
for rp in "${rp_list[@]}"; do
      sbatch 10_SubmitterATOS_prob_ff_meteo_verif_short_fc.sh $rp
done