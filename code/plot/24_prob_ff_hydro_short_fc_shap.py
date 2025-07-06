import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

#########################################################################################################
# CODE DESCRIPTION
# 24_prob_ff_hydro_short_fc_shap.py computes the shap values for the short-range model predictions.

# Usage: python3 24_prob_ff_hydro_short_fc_shap.py

# Runtime: ~ 5 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# feature_cols (list of strings): list of feature columns' names, i.e. model's predictors.
# target_col (string): target column's name, i.e. model's predictand.
# model_name (string): name of the model to train.
# loss_func_list (list of strings): type of loss function considered. Valid values are:
#                                                           - bce: no weights applied to loss function.
#                                                           - weighted_bce: wheight applied to loss function.
# eval_metric_list (list of strings): evaluation metric for the data-driven models. Valid values are:
#                                                           - auc: area under the roc curve.
#                                                           - auprc: area under the precion-recall curve.
# git_repo (string): repository's local path.
# file_in_pdt (string): relative path of the file containing the point data table to consider.
# dir_in (string): relative path of the directory containing the shap values.
# dir_out (string): relative path of the directory containing the shap plots.

##################################################################################################
# INPUT PARAMETERS
feature_cols = ["tp_prob_1", "tp_prob_max_1_adj_gb", "tp_prob_50", "tp_prob_max_50_adj_gb", "swvl", "sdfor", "lai"]
target_col = "ff"
model_name = "gradient_boosting_xgboost"
loss_func_list = ["bce", "weighted_bce"]
eval_metric_list = ["auc", "auprc"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_pdt = "data/processed/11_prob_ff_hydro_short_fc_combine_pdt/pdt_2021_2024.csv"
dir_in = "data/processed/14_prob_ff_hydro_short_fc_shap"
dir_out = "data/plot/24_prob_ff_hydro_short_fc_shap"
##################################################################################################


# Uploading the test dataset
print("Reading the predictors")
df = pd.read_csv(f"{git_repo}/{file_in_pdt}")
X = df[feature_cols].copy()


# Creating the verification plots
for loss_func in ["bce", "weighted_bce"]:

      for eval_metric in ["auc", "auprc"]:

            # Creating output directory
            dir_out_temp = f"{git_repo}/{dir_out}/{loss_func}/{eval_metric}/{model_name}"
            os.makedirs(dir_out_temp, exist_ok=True)

            # Reading the SHAP values
            print(f"Creating SHAP plots for {loss_func} and {eval_metric}")
            file_in = f"{git_repo}/{dir_in}/{loss_func}/{eval_metric}/shap_{model_name}.pkl"
            shap_values = joblib.load(file_in)

            # Creating and saving the SHAP plot for global feature importance
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/global_feature_importance.png", dpi=1000, bbox_inches="tight")
            plt.close()

            # Creating and saving the SHAP dependence plots
            shap.dependence_plot("tp_prob_1", shap_values, X, interaction_index="swvl", dot_size=5, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/tp_prob_1_DEP_swvl.png", dpi=1000)
            plt.close()

            shap.dependence_plot("tp_prob_1", shap_values, X, interaction_index="sdfor", dot_size=5, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/tp_prob_1_DEP_sdfor.png", dpi=1000)
            plt.close()

            shap.dependence_plot("tp_prob_1", shap_values, X, interaction_index="lai", dot_size=5, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/tp_prob_1_DEP_lai.png", dpi=1000)
            plt.close()

            shap.dependence_plot("tp_prob_50", shap_values, X, interaction_index="swvl", dot_size=5, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/tp_prob_50_DEP_swvl.png", dpi=1000)
            plt.close()

            shap.dependence_plot("tp_prob_50", shap_values, X, interaction_index="sdfor", dot_size=5, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/tp_prob_50_DEP_sdfor.png", dpi=1000)
            plt.close()

            shap.dependence_plot("tp_prob_50", shap_values, X, interaction_index="lai", dot_size=5, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_out_temp}/tp_prob_50_DEP_lai.png", dpi=1000)
            plt.close()