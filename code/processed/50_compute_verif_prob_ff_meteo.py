import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import metview as mv
from sklearn.metrics import roc_auc_score, confusion_matrix

####################################################################################################################
# CODE DESCRIPTION
# 50_compute_verifi_prob_ff_meteo_fc.py computes the forecasts probabilities of having a flash flood event.
# Runtime: the code takes up to 12 hours.

# INPUT PARAMETERS DESCRIPTION
# base_date_s (date, in YYYYMMDD format): start base date to consider in the verification period.
# base_date_f (date, in YYYYMMDD format): fina base date to consider in the verification period.
# step_f_start (integer, in hours): first final-step of the accumulation period.
# step_f_final (integer, in hours): last final-step of the accumulation period.
# disc_step (integer, in hours): step discretisation.
# rp_list (list of integers): list of rainfall thresholds expressed as return periods (in years).
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in_ff (string): relative path of the directory containing the flash flood observations. 
# dir_in_prob_tp (string): relative path of the directory containing the rainfall probabilities of exceeding a certain return period.
# dir_in_prob_tp_fc (string): relative path of the directory containing the forecasts of rainfall probabilities of exceeding a certain return period.
# dir_out (string): relative path of the directory containing the instances of yes-events.

# INPUT PARAMETERS
base_date_s = datetime(2024,1,1,0)
base_date_f = datetime(2024,12,31,0)
step_f_start = 24
step_f_final = 120
disc_step = 24
rp_list = [1, 5, 10, 20, 50, 100]
num_bs = 100
alpha = 95
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_era5/mask.grib"
dir_in_ff = "data/compute/15_grid_acc_reports_ff"
dir_in_prob_tp = "data/compute/08_tp_prob_exceed_rp"
dir_in_prob_tp_fc = "data/compute/40_tp_prob_exceed_rp_fc"
dir_out = "data/compute/30_prob_ff_meteo_fc"
####################################################################################################################


# Set the rainfall accumulation period
acc = 24

# Reading the domain's mask
mask = mv.values(mv.read(git_repo + "/" + file_in_mask))
ind_mask = np.where(mask == 1)[0]


#############
# REANALYSIS #
#############

print(f' *** VERIFICATION CONSIDERING RAINFALL REANALYSIS ***')

aroc_all = []
aroc_ci_l = []
aroc_ci_h = []
fb_all = []
for rp in rp_list:

      print(f'\nVerifying flash floood events considering rainfall exceeding {rp}-year return period')
      dates_all = []
      prob_all = []
      ff_all = []
      aroc_bs = []
      fb_bs = []
      the_date_s = base_date_s
      while the_date_s <= base_date_f:

            the_date_f = the_date_s + timedelta(hours = acc)
            dates_all.append(the_date_f)
            print(f' - Reading rainfall reanalysis and flash flood reports valid for the {acc}-hourly accumulation period ending on {the_date_f.strftime("%Y%m%d")} at {the_date_f.strftime("%H")} UTC')
            
            file_prob = f'{git_repo}/{dir_in_prob_tp}/{rp}rp/{the_date_f.strftime("%Y%m")}/prob_exceed_rp_{the_date_f.strftime("%Y%m%d")}_00.grib'
            prob = mv.values(mv.read(file_prob))[ind_mask]
            prob_all.append(prob)

            file_ff = f'{git_repo}/{dir_in_ff}/{the_date_f.strftime("%Y")}/grid_acc_reports_ff_{the_date_f.strftime("%Y%m%d")}_{the_date_f.strftime("%H")}.grib'
            if os.path.exists(file_ff) is True:
                  ff = mv.values(mv.read(file_ff)>0)[ind_mask]
            else:
                  ff = mv.values(mv.read(file_prob)*0)[ind_mask] # no flash floods reported on the considered day
            ff_all.append(ff)

            the_date_s = the_date_s + timedelta(days = 1)
      
      prob_all = np.array(prob_all)
      ff_all = np.array(ff_all)
      m = prob_all.shape[0]

      aroc_all.append(roc_auc_score(ff_all.ravel(), prob_all.ravel()))
      # tn, fp, fn, tp = confusion_matrix(ff_all, prob_all>0).ravel()
      # fb_all.append((tp + fp) / (tp + fn))

      ind_bs_list = np.random.randint(0, m-1, size=(num_bs, m))
      for i in range(num_bs):
            ind_bs = ind_bs_list[i,:]
            prob_bs_temp = prob_all[ind_bs, :].ravel()
            ff_bs_temp = ff_all[ind_bs, :].ravel()
            aroc_bs.append(roc_auc_score(ff_bs_temp, prob_bs_temp))
            # tn, fp, fn, tp = confusion_matrix(ff_bs_temp, prob_bs_temp>0)
            # fb_bs.append((tp + fp) / (tp + fn))

      aroc_ci_l.append(np.percentile(aroc_bs, ((100 - alpha) / 2)))
      aroc_ci_h.append(np.percentile(aroc_bs, ((100 + alpha) / 2)))

print(aroc_all)
print(aroc_ci_l)
print(aroc_ci_h)
exit()



############
# FORECASTS #
############


# Reading the rainfall forecasts and the flash flood observations
base_date = base_date_s
while base_date <= base_date_f:
      
      prob_ff = pd.DataFrame()
      for step_f in range(step_f_start, step_f_final + 1, disc_step):
      
            step_s = step_f - acc

            # Reading the point-rainfall forecasts
            print(f'\nReading point-rainfall forecast for {base_date.strftime("%Y%m%d")} at {base_date.strftime("%H")} UTC (t+{step_s}, t+{step_f})')
            file_tp = f'{git_repo}/{dir_in_prob_tp_fc}/{base_date.strftime("%Y%m")}/Pt_BC_PERC_{base_date.strftime("%Y%m%d")}_{step_f:03d}.grib2'
            tp = mv.read(file_tp)
            n = int(mv.count(tp))

            # Reading the flash flood observations for the corresponding valid time
            vt_s = base_date + timedelta(hours = step_s)
            vt_f = base_date + timedelta(hours = step_f)
            print(f' - Reading the corresponding flash flood observations accumulated between {vt_s.strftime("%Y%m%d")} at {vt_s.strftime("%H")} UTC and {vt_f.strftime("%Y%m%d")} at {vt_f.strftime("%H")} UTC')
            file_ff = f'{git_repo}/{dir_in_ff}/{vt_s.strftime("%Y")}/grid_acc_reports_ff_{vt_s.strftime("%Y%m%d")}_{vt_s.strftime("%H")}.grib'
            if os.path.exists(file_ff) is True:
                  ff = mv.read(file_ff)
            else:
                  ff = tp[0] * 0 # no flash floods reported on the considered day
            
            ff_mask = mv.values(ff)[ind_mask]
            prob_ff["ff"] = ff_mask

            # Computing the number of ensemble members exceeding the considered return period
            print(f' - Computing the probabilities of exceeding a certain rainfall threshold')
            for rp_2_compute in rp_2_compute_list:
                  ind_rp = np.where(rp_list == rp_2_compute)[0]
                  climate_rp = tp_climate[ind_rp]
                  prob_ff_rp = (mv.values(mv.sum(tp >= mv.duplicate(climate_rp, n)) / n * 100))[ind_mask]
                  prob_ff[f"rp_{rp_2_compute}"] = prob_ff_rp

            # Save the dataframe
            dir_out_temp = f'{git_repo}/{dir_out}/{base_date.strftime("%Y%m")}'
            os.makedirs(dir_out_temp, exist_ok=True)
            prob_ff.to_csv(f'{dir_out_temp}/prob_ff_{base_date.strftime("%Y%m%d")}_{base_date.strftime("%H")}_{step_f:03d}.csv', index=False)
      
      base_date = base_date + timedelta(days = 1)

































# Reading the rainfall forecasts and the flash flood observations
base_date = base_date_s
while base_date <= base_date_f:
      
      prob_ff = pd.DataFrame()
      for step_f in range(step_f_start, step_f_final + 1, disc_step):
      
            step_s = step_f - acc

            # Reading the point-rainfall forecasts
            print(f'\nReading point-rainfall forecast for {base_date.strftime("%Y%m%d")} at {base_date.strftime("%H")} UTC (t+{step_s}, t+{step_f})')
            file_tp = f'{git_repo}/{dir_in_tp_fc}/{base_date.strftime("%Y%m")}/Pt_BC_PERC_{base_date.strftime("%Y%m%d")}_{step_f:03d}.grib2'
            tp = mv.read(file_tp)
            n = int(mv.count(tp))

            # Reading the flash flood observations for the corresponding valid time
            vt_s = base_date + timedelta(hours = step_s)
            vt_f = base_date + timedelta(hours = step_f)
            print(f' - Reading the corresponding flash flood observations accumulated between {vt_s.strftime("%Y%m%d")} at {vt_s.strftime("%H")} UTC and {vt_f.strftime("%Y%m%d")} at {vt_f.strftime("%H")} UTC')
            file_ff = f'{git_repo}/{dir_in_ff}/{vt_s.strftime("%Y")}/grid_acc_reports_ff_{vt_s.strftime("%Y%m%d")}_{vt_s.strftime("%H")}.grib'
            if os.path.exists(file_ff) is True:
                  ff = mv.read(file_ff)
            else:
                  ff = tp[0] * 0 # no flash floods reported on the considered day
            
            ff_mask = mv.values(ff)[ind_mask]
            prob_ff["ff"] = ff_mask

            # Computing the number of ensemble members exceeding the considered return period
            print(f' - Computing the probabilities of exceeding a certain rainfall threshold')
            for rp_2_compute in rp_2_compute_list:
                  ind_rp = np.where(rp_list == rp_2_compute)[0]
                  climate_rp = tp_climate[ind_rp]
                  prob_ff_rp = (mv.values(mv.sum(tp >= mv.duplicate(climate_rp, n)) / n * 100))[ind_mask]
                  prob_ff[f"rp_{rp_2_compute}"] = prob_ff_rp

            # Save the dataframe
            dir_out_temp = f'{git_repo}/{dir_out}/{base_date.strftime("%Y%m")}'
            os.makedirs(dir_out_temp, exist_ok=True)
            prob_ff.to_csv(f'{dir_out_temp}/prob_ff_{base_date.strftime("%Y%m%d")}_{base_date.strftime("%H")}_{step_f:03d}.csv', index=False)
      
      base_date = base_date + timedelta(days = 1)