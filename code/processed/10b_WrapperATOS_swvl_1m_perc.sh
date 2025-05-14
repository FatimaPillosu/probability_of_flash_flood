#!/bin/bash

Year_S=2001
Year_F=2004
echo "Computing the water content in the soil (in percentage) for year:"
for Year in $(seq $Year_S $Year_F); do
      echo " - $Year"
      sbatch 10a_SubmitterATOS_swvl_1m_perc.sh $Year
done