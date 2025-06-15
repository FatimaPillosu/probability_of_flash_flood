#!/bin/bash

model_2_train_list=("gradient_boosting_xgboost" "random_forest_xgboost" "gradient_boosting_catboost" "gradient_boosting_lightgbm" "random_forest_lightgbm" "feed_forward_keras")
loss_fn_choice_list=("bce" "weighted_bce")
eval_metric_list=("auc" "auprc")

for model_2_train in "${model_2_train_list[@]}"; do
      for loss_fn_choice in "${loss_fn_choice_list[@]}"; do
            for eval_metric in "${eval_metric_list[@]}"; do
                  echo " "
                  echo "Model: ${model_2_train},  Loss function: ${loss_fn_choice}, Evaluation metric: ${eval_metric}"
                  sbatch 12a_SubmitterATOS_prof_ff_hydro_short_fc_train_ml_cv_optuna.sh ${model_2_train} ${loss_fn_choice} ${eval_metric}
            done
      done
done