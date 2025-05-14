#!/bin/bash

Year_S=2001
Year_F=2024
echo "Computing the probabilities of exceeding a certain return-period for ERA5-ecPoint's 24-hourly rainfall:"
for Year in $(seq $Year_S $Year_F); do
      echo " - $Year"
      sbatch 08a_SubmitterATOS_tp_prob_exceed_rp.sh $Year
done