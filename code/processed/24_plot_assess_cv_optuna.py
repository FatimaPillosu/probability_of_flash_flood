import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

#################################################################################
# CODE DESCRIPTION
# 24_plot_assess_cv_optuna.py cretes the plot to assess which hyperparameter combination yields 
# the best mean cross validation scores, defining the generealisation performance of the considered 
# machine learning models.
# Runtime: negligible.

# INPUT PARAMETERS DESCRIPTION
# num_k_folds (integer): number of k-folds considered in the stratified k-fold cross validation.
# ml_model_list (list of strings): names of the considered machine learning models. Valid values are:
#                                                           - random_forest_xgboost
#                                                           - random_forest_lightgbm
#                                                           - gradient_boosting_xgboost
#                                                           - gradient_boosting_lightgbm
#                                                           - gradient_boosting_catboost
#                                                           - gradient_boosting_adaboost
#                                                           - feed_forward_keras
# git_repo (string): repository's local path.
# file_in (string): relative path of the file containing the training dataset.
# dir_out (string): relative path of the directory containing the trained machine learning models.

# INPUT PARAMETERS
num_k_folds = 5
ml_model_list = ['gradient_boosting_xgboost', 'random_forest_xgboost', 'random_forest_lightgbm', 'gradient_boosting_lightgbm', 'gradient_boosting_catboost', 'feed_forward_keras']
feature_cols = ["tp_prob_1", "tp_prob_50", "swvl", "sdfor", "lai"] 
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in = "data/compute/23_train_ml_cv_optuna_old"
dir_out = "data/compute/24_dt_cv_optuna"
#################################################################################


# Create labels for bar plots
labels = []
for k_fold in range(num_k_folds):
    labels.append(f"kfold_{k_fold+1}")

# Plotting the single scores from the stratified k-fold cross validation
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ml_model in ml_model_list:
    recall = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/recall.npy")
    f1 = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/f1.npy")
    aroc = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/aroc.npy")
    

    axes[0,0].plot(labels, recall, label = ml_model)
    axes[0,0].set_title("Recall")
    axes[0,0].legend()
    axes[1,0].plot(labels, f1, label = ml_model)
    axes[1,0].set_title("F1 score")
    axes[0,1].plot(labels, aroc, label = ml_model)
    axes[0,1].set_title("Aroc")
plt.show()
# plt.close()

# # Plot the roc curves
# for ml_model in ml_model_list:
#     aroc = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/aroc.npy")
#     for k_fold in range(1, num_k_folds+1): 
#         far = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/far_" + str(k_fold) + ".npy")
#         hr = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/hr_" + str(k_fold) + ".npy")
#         plt.plot(far, hr, lw=0.5, label="AROC_" + str(k_fold) + "_fold = %0.5f" % aroc[k_fold-1])
#         plt.plot([-0.01, 1.01], [-0.01, 1.01], linestyle="-", color="gray")  
#         plt.xlim([-0.01, 1.01])
#         plt.ylim([-0.01, 1.01])
#         plt.xlabel("False Alarm Rate")
#         plt.ylabel("Hit Rate")
#         plt.title("Receiver Operating Characteristic (ROC) Curve - " + ml_model)
#         plt.legend(loc="lower right")
#     plt.close()

# # Plot the reliability diagrams
# for ml_model in ml_model_list:
#     for k_fold in range(1, num_k_folds+1): 
#         fc_pred = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/fc_pred_" + str(k_fold) + ".npy")
#         obs_freq = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/obs_freq_" + str(k_fold) + ".npy")
#         plt.plot(fc_pred, obs_freq, lw = 0.5, marker=".", linestyle="-")
#         plt.plot([0, 1], [0, 1], linestyle="-", color="gray")
#         plt.xlabel("Forecast probability")
#         plt.ylabel("Observed frequency")
#         plt.title("Reliability Diagram - " + ml_model)
#     plt.show()

# # Print feature importance
# for ml_model in ml_model_list:
    
#     for k_fold in range(1, num_k_folds+1): 
        
#         model = joblib.load(git_repo + "/" + dir_in + "/" + ml_model + "/model_" + str(k_fold) + ".joblib")
#         booster = model.get_booster()
#         gain_importance_dict = booster.get_score(importance_type='gain')
#         split_importance_dict = booster.get_score(importance_type='weight')
#         print(split_importance_dict)
#         exit()

#         print()
#         for f in feature_cols:
#             gain = gain_importance_dict.get(f, 0.0)
#             split = split_importance_dict.get(f, 0)
#             print(f"{f:<10} | Gain: {gain:>10.2f} | Splits: {split}")