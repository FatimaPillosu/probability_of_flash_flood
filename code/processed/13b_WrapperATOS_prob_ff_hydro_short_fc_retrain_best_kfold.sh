#!/bin/bash

model_2_train_list=("feed_forward_keras")
loss_fn_choice_list=("weighted_bce")
eval_metric_list=("auc" "auprc")
rep_to_run=1
outer_fold_to_run_list=(1 5)

for model_2_train in "${model_2_train_list[@]}"; do
      for loss_fn_choice in "${loss_fn_choice_list[@]}"; do
            for ((i=0; i<${#eval_metric_list[@]}; i++)); do
                  eval_metric=${eval_metric_list[$i]}
                  outer_fold_to_run=${outer_fold_to_run_list[$i]}
                  echo " "
                  echo ${model_2_train} ${loss_fn_choice} ${eval_metric} ${rep_to_run} ${outer_fold_to_run}
                  echo "Model: ${model_2_train},  Loss function: ${loss_fn_choice}, Evaluation metric: ${eval_metric}"
                  sbatch 13a_SubmitterATOS_prob_ff_hydro_short_fc_retrain_best_kfold.sh ${model_2_train} ${loss_fn_choice} ${eval_metric} ${rep_to_run} ${outer_fold_to_run}
            done
      done
done