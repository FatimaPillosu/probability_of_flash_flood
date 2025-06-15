import numpy as np

file = "/home/mofp/vol_ecpoint_dev/mofp/phd/probability_of_flash_flood/data/processed/12_prob_ff_hydro_short_fc_train_ml_cv_optuna/auc/gradient_boosting_xgboost/best_threshold.npy"
a = np.load(file)
print(a)