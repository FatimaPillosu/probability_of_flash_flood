#!/bin/bash

rp_list=(1 5 10 20)
for rp in "${rp_list[@]}"; do
      sbatch 11_SubmitterATOS_prob_ff_meteo_verif_long_fc.sh $rp
done