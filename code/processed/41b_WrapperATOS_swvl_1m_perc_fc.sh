#!/bin/bash

Year_S=2021
Year_F=2024
echo "Computing the water content in the soil (in percentage) for year:"
for Year in $(seq $Year_S $Year_F); do
      echo " - $Year"
      sbatch 41a_SubmitterATOS_swvl_1m_perc_fc.sh $Year
done