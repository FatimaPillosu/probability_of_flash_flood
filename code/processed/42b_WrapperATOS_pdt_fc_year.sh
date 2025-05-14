#!/bin/bash

year_s=2021
year_f=2024
echo "Computing PDT for year:"
for year in $(seq $year_s $year_f); do
      echo " - $year"
      sbatch 42a_SubmitterATOS_pdt_fc_year.sh ${year}
done