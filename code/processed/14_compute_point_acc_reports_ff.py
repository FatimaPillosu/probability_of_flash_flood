import os
from datetime import datetime, timedelta
import pandas as pd

##################################################################################
# CODE DESCRIPTION
# 14_compute_point_acc_reports_ff.py creates the point accumulated flash flood reports over the 
# same rainfall's 24-hourly accumulation periods.
# Runtime: the code takes up to 2 minutes.

# INPUT PARAMETERS DESCRIPTION
# year_s (integer): start year to consider.
# year_f (integer): final year to consider.
# git_repo (string): repository's local path.
# file_in (string): relative path of the file containing the flash flood reports.
# dir_out (string): relative path of the directory containing the point accumulated flash flood reports.

# INPUT PARAMETERS
year_s = 1950
year_f = 2024
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in = "data/compute/12_extract_noaa_reports_ff/noaa_reports_ff.csv"
dir_out = "data/compute/14_point_acc_reports_ff"
##################################################################################


# Defining the accumulation periods to consider
the_date_start_s = datetime(year_s,1,1)
the_date_start_f = datetime(year_f,12,31)

# Reading the flash flood reports
ff = pd.read_csv(git_repo + "/" + file_in)
ff["REPORT_DATE"] = pd.to_datetime(ff["REPORT_DATE"])

# Accumulating the point flash flood reports
the_date_start = the_date_start_s
while the_date_start <= the_date_start_f:
      
      print()
      the_date_final = the_date_start + timedelta(hours=24)
      print("Accumulating point flood reports between " + the_date_start.strftime("%Y-%m-%d") + " at " + the_date_start.strftime("%H") + " UTC and " + the_date_final.strftime("%Y-%m-%d") + " at " + the_date_final.strftime("%H") + " UTC")
      
      # Select the flash flood reports for the specific accumulation period
      filtered_ff = ff[(ff["REPORT_DATE"] >= the_date_start) & (ff["REPORT_DATE"] < the_date_final)]

      # Saving the filtered files
      filtered_ff_size = len(filtered_ff)
      if filtered_ff_size != 0:
            print(" - Saving " + str(filtered_ff_size) + " reports")
            dir_out_temp = git_repo + "/" + dir_out + "/" + the_date_final.strftime("%Y")
            if not os.path.exists(dir_out_temp):
                  os.makedirs(dir_out_temp)
            file_out = dir_out_temp + "/point_acc_reports_ff_" + the_date_final.strftime("%Y%m%d%H") + ".csv"
            filtered_ff.to_csv(file_out, index=False)
      else: 
            print(" - No reports to save")
      
      the_date_start = the_date_start + timedelta(hours=24)