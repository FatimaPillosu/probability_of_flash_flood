#!/bin/bash

model_2_train_list=("gradient_boosting_xgboost" "random_forest_xgboost" "gradient_boosting_catboost" "gradient_boosting_lightgbm" "random_forest_lightgbm" "feed_forward_keras" "gradient_boosting_adaboost")

echo "Training the model:"
for model_2_train in "${model_2_train_list[@]}"; do
      echo " "
      echo ${model_2_train}
      sbatch 12a_SubmitterATOS_prof_ff_hydro_short_fc_train_ml_cv_optuna.sh ${model_2_train}
done