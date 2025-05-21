import os
import numpy as np
import matplotlib.pyplot as plt

#######################################################################
# CODE DESCRIPTION
# 09_timeseries_counts_noaa_reports_ff.py plots the yearly timeseries for the counts 
# of flood reports in the NOAA database, comparing the counts of all reports and only 
# flash flood ones. 

# Usage: python3 09_timeseries_counts_noaa_reports_ff.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the flash flood reports.
# dir_out (string): relative path of the directory containing the timeseries plot.

#######################################################################
# INPUT PARAMETERS
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/01_extract_noaa_reports_ff"
dir_out = "data/plot/09_timeseries_counts_noaa_reports_ff"
#######################################################################

# Read the data to plot
temp_dir_in = git_repo + "/" + dir_in
years_rep = np.load(temp_dir_in + "/years.npy")
num_rep_all = np.load(temp_dir_in + "/counts_reports_all.npy")
num_rep_ff = np.load(temp_dir_in + "/counts_reports_ff.npy")
num_rep_ff_withCoord = np.load(temp_dir_in + "/counts_reports_ff_with_coord.npy")

# Plot the counts
fig, ax = plt.subplots(figsize=(10, 8))
index = np.arange(len(years_rep))
bar_width = 0.7
opacity = 1

rects1 = ax.bar(years_rep, num_rep_all, bar_width, alpha=opacity, color="gainsboro", align='center', label="All flood types")
rects2 = ax.bar(years_rep, num_rep_ff, bar_width, alpha=opacity, color="maroon", align='center', label="Only flash floods")
rects3 = ax.bar(years_rep, num_rep_ff_withCoord, bar_width, alpha=opacity, color="red", align='center', label="Only flash floods with lat/lon coordinates")

ax.set_xlabel("Years", fontsize=16, labelpad = 10)
ax.set_ylabel("Counts", fontsize=16, labelpad = 10)
ax.set_title("Count of flood reports per year", fontsize=18, pad=15, weight = "bold")
ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=16)

# Saving the plots
main_dir_out = git_repo + "/" + dir_out
if not os.path.exists(main_dir_out):
    os.makedirs(main_dir_out)
file_out = main_dir_out + "/timeseries_counts_noaa_reports_ff.jpeg" 
plt.savefig(file_out, format="jpeg", bbox_inches="tight", dpi=300)