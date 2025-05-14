import os
import numpy as np
import matplotlib.pyplot as plt

#################################################################################
# CODE DESCRIPTION
# 26_plot_assess_retrain_ml.py creates the plot to assess the final versions of the re-train models.
# Runtime: negligible.

# INPUT PARAMETERS DESCRIPTION
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
ml_model_list = ["random_forest_xgboost", "random_forest_lightgbm", "gradient_boosting_xgboost", "gradient_boosting_lightgbm", "gradient_boosting_catboost"] 
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
dir_in = "data/compute/25_retrain_ml_test"
dir_out = "data/plot/26_assess_retrain_ml"
#################################################################################


# Create the output directory
dir_out_temp = git_repo + "/" + dir_out
os.makedirs(dir_out_temp, exist_ok=True)

# Plotting the assessment of the final trained models
recall = []
f1 = []
best_threshold = []
for ml_model in ml_model_list:

    print()
    print(f"Examining model: {ml_model}")
    test_scores = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/test_scores.npy")
    recall.append(test_scores[0])
    f1.append(test_scores[1])
    best_threshold.append(test_scores[3]*100)
    aroc = test_scores[2]

    #Plotting the reliability diagrams
    fc_pred = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/fc_pred.npy")
    obs_freq = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/obs_freq.npy")
    plt.figure(figsize=(10, 6))
    plt.plot(fc_pred, obs_freq, lw = 0.5, marker=".", linestyle="-")
    plt.plot([0, 1], [0, 1], linestyle="-", color="gray")
    plt.xlabel("Forecast probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Diagram - " + ml_model)
    plt.savefig(dir_out_temp + "/rel_diag_" + ml_model + ".png")
    plt.close()

    # Plotting the roc curves
    far = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/far.npy")
    hr = np.load(git_repo + "/" + dir_in + "/" + ml_model + "/hr.npy")
    plt.figure(figsize=(10, 6))
    plt.plot(far, hr, lw=0.5, label="AROC = %0.5f" % aroc)
    plt.plot([-0.01, 1.01], [-0.01, 1.01], linestyle="-", color="gray")  
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Hit Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(dir_out_temp + "/roc_" + ml_model + ".png")
    plt.close()

# Plotting recall 
plt.figure(figsize=(8, 8))
plt.bar(ml_model_list, recall)
plt.xticks(rotation = 30, fontsize=6)
plt.savefig(dir_out_temp + "/recall.png")
plt.close()

# Plotting f1 
plt.figure(figsize=(8, 8))
plt.bar(ml_model_list, recall)
plt.xticks(rotation = 30, fontsize=6)
plt.savefig(dir_out_temp + "/f1.png")
plt.close()