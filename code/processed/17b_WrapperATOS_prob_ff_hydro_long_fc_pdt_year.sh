#!/bin/bash

year_s=2001
year_f=2024
echo "Computing PDT for year:"
for year in $(seq $year_s $year_f); do
      echo " - $year"
      sbatch 17a_SubmitterATOS_prob_ff_hydro_long_fc_pdt_year.sh ${year}
done