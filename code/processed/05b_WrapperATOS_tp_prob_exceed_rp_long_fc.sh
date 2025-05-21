#!/bin/bash

Year_S=2021
Year_F=2024
echo "Computing the forecasts probabilities of exceeding a certain return-period for ERA5-ecPoint's 24-hourly long-range rainfall:"
for Year in $(seq $Year_S $Year_F); do
      echo " - $Year"
      sbatch 05a_SubmitterATOS_tp_prob_exceed_rp_long_fc.sh $Year
done