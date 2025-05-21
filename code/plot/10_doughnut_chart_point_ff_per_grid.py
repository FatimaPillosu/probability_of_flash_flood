import os
from datetime import datetime, timedelta
import numpy as np
import metview as mv
import matplotlib.pyplot as plt

#################################################################################################
# CODE DESCRIPTION
# 10_doughnut_chart_point_ff_per_grid.py plots a doughnut chart of the distribution of the accumulated point flash 
# flood reports per grid-box in the north-west (NW), north-east(NE), south-west (SW) and south-east (SE) of the US.

# Usage: python3 c.py

# Runtime: ~ 5 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year_s (integer): start year to consider.
# year_f (integer): final year to consider.
# north_south_lat_boundary (integer, from -90 to +90): value of the latitude boundary that separates north from south.
# west_east_lon_boundary (integer, from -180 to +180): value of the longitude boundary that separates west from east.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the accumulated point flash flood reports per grid-box.
# dir_out (string): relative path of the directory containing the doughnut chart.

#################################################################################################
# INPUT PARAMETERS
year_s = 2001
year_f= 2024
north_south_lat_boundary = 38
west_east_lon_boundary = -100
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/03_grid_acc_reports_ff"
dir_out = "data/plot//ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
#################################################################################################


print()
print("Plotting the doughnut chart of the distribution of the accumulated gridded flash flood reports in the north-west (NW), north-east(NE), south-west (SW) and south-east (SE) of the US")

# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)
mask = mv.bitmap(mask,0) # bitmap the values outside the domain

# Defining the accumulation periods to consider
the_date_start_s = datetime(year_s,1,1)
the_date_start_f = datetime(year_f,12,31)

# Initializing the variable that will store the absolute frequency of accumulated gridded flash flood reports per grid-box within the considered period
abs_freq_grid_ff = 0

# Adding up the number of accumulated gridded flash flood reports observed within the considered period
print()
print("Computing the relative frequency of 24-hourly gridded flash flood reports in each grid-box of the domain between " + str(year_s) + " and " + str(year_f))
print("Adding the reports in the period ending:")
the_date_start = the_date_start_s
while the_date_start <= the_date_start_f:
      the_date_final = the_date_start + timedelta(hours=24)
      print(" - on " + the_date_final.strftime("%Y-%m-%d") + " at " + the_date_final.strftime("%H") + " UTC")
      file_in_grid_ff_single_day = git_repo + "/" + dir_in + "/" + the_date_final.strftime("%Y") + "/grid_acc_reports_ff_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
      if os.path.exists(file_in_grid_ff_single_day):
            grid_ff_single_day = mv.read(file_in_grid_ff_single_day)
            abs_freq_grid_ff = abs_freq_grid_ff + grid_ff_single_day
      the_date_start = the_date_start + timedelta(hours=24)
abs_freq_grid_ff = abs_freq_grid_ff * mask

# Defining the count of accumulated gridded flash flood reports in four regions of US (i.e. NW, NE, SW, and SE)
lats = mv.latitudes(abs_freq_grid_ff)
lons = mv.longitudes(abs_freq_grid_ff) - 360
mask_nw = np.where((lats >= north_south_lat_boundary) & (lons <= west_east_lon_boundary))[0]
mask_ne = np.where((lats >= north_south_lat_boundary) & (lons > west_east_lon_boundary))[0]
mask_sw = np.where((lats < north_south_lat_boundary) & (lons <= west_east_lon_boundary))[0]
mask_se = np.where((lats < north_south_lat_boundary) & (lons > west_east_lon_boundary))[0]
abs_freq_grid_ff_vec = mv.values(abs_freq_grid_ff)
abs_freq_grid_ff_nw = np.nansum(abs_freq_grid_ff_vec[mask_nw])
abs_freq_grid_ff_ne = np.nansum(abs_freq_grid_ff_vec[mask_ne])
abs_freq_grid_ff_sw = np.nansum(abs_freq_grid_ff_vec[mask_sw])
abs_freq_grid_ff_se = np.nansum(abs_freq_grid_ff_vec[mask_se])

# Creating the doughnut chart
labels = ["NW", "SW", "SE", "NE"]
values = [abs_freq_grid_ff_nw, abs_freq_grid_ff_sw, abs_freq_grid_ff_se, abs_freq_grid_ff_ne]
colours = ["#EEA320", "#B4BC3D", "#00A4DC", "#1CB8A6"]
explode = (0.01, 0.01, 0.01, 0.01)

sum_values = sum(values)
perc_values = values/sum_values*100
print()
print("Total number of accumulated gridded flash flood reports in the US = " + str(int(sum_values)))
print("Percentages per regions:")
for i in range(len(perc_values)):
      print(labels[i] + " = " + str(np.round(perc_values[i], decimals=1)) + "%")

plt.pie(values, colors=colours, labels=labels, startangle=90, pctdistance=0.85, explode=explode, textprops={'fontsize': 16})
centre_circle = plt.Circle((0, 0), 0.50, color='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.text(0, 0, f'Total\n{int(sum_values)}', ha='center', va='center', fontsize=16)

# Saving the doughnut chart
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
    os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/doughnut_chart_point_ff_per_grid.jpeg" 
plt.savefig(file_out, format="jpeg", bbox_inches="tight", dpi=5000)