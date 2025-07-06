#!/bin/bash

year_s=2021
year_f=2024
step_f_start=24
step_f_final=120
for step_f in $(seq ${step_f_start} 24 ${step_f_final}); do
      for year in $(seq $year_s $year_f); do
            echo "Computing the PDT for t+${step_f} and ${year}"
            sbatch 15a_SubmitterATOS_prob_ff_hydro_long_fc_pdt_year.sh ${year} ${step_f}
      done
done