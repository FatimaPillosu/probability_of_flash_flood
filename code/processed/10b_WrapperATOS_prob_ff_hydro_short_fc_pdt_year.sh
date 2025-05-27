#!/bin/bash

year_s=2001
year_f=2024
echo "Computing PDT for year:"
for year in $(seq $year_s $year_f); do
      echo " - $year"
      sbatch 10a_SubmitterATOSprob_ff_hydro_short_fc_pdt_year.sh ${year}
done