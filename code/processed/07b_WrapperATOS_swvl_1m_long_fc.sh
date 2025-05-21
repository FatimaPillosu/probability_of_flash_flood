#!/bin/bash

Year_S=2021
Year_F=2024
echo "Computing the water content in the soil (in percentage) for year:"
for Year in $(seq $Year_S $Year_F); do
      echo " - $Year"
      sbatch 07a_SubmitterATOS_swvl_1m_long_fc.sh $Year
done