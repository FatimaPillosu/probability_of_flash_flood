import os
from datetime import datetime, timedelta
import metview as mv

##################################################################################
# CODE DESCRIPTION
# 15_tp_prob_exceed_rp_long_fc.py computes the probabilities of 24-hourly long-range rainfall from 
# ERA5-ecPoint to exceed the considered return period.

# Usage: python3 15_tp_prob_exceed_rp_long_fc.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# base_date (date, in YYYYMMDDHH format): base date to consider, including run time (given by HH).
# step_f (integer, in hours): final step of the considered accumulation period.
# rp_list (list of integers): list of the return periods.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the probabilities.
# dir_out (string): relative path of the directory containing the probability's plots.

##################################################################################
# INPUT PARAMETERS
base_date = datetime(2021,8,28,0)
step_f = 120
rp_list = [1, 2, 5, 10, 20, 50, 100]
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/05_tp_prob_exceed_rp_long_fc"
dir_out = "data/plot/15_tp_prob_exceed_rp_long_fc"
##################################################################################


# Defining the accumulation period to consider
acc = 24
step_s = step_f - acc

# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)
mask_vals = mv.values(mask)
mask_bitmap = mv.bitmap(mask, 0)

# Plotting the probabilities of ERA5_ecPoint's 24-hourly long-range rainfall forecasts
print()
print(f'Considering ERA5_ecPoint 24-hourly long-range rainfall forecasts for {base_date.strftime("%Y%m%d")} at {base_date.strftime("%H")} UTC (t+{step_s}, t+{step_f})')
print("Plotting the probabilities exceeding:")
for rp in rp_list:

      print(" - " + str(rp) + "-year return period")

      # Reading the probabilities
      prob = mv.read(git_repo + "/" + dir_in + "/" + str(rp) + "rp" + "/" + base_date.strftime("%Y%m") + "/prob_exceed_rp_" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}" + ".grib")
      prob_bitmap = mv.bitmap(prob, mask_bitmap)
      
      # Defining the forecasts' valid times and related plot titles
      ValidityDateS = base_date +  timedelta(hours=step_s)
      DayVS = ValidityDateS.strftime("%d")
      MonthVS = ValidityDateS.strftime("%B")
      yearVS = ValidityDateS.strftime("%Y")
      TimeVS = ValidityDateS.strftime("%H")

      ValidityDateF = base_date +  timedelta(hours=step_f)
      DayVF = ValidityDateF.strftime("%d")
      MonthVF = ValidityDateF.strftime("%B")
      yearVF = ValidityDateF.strftime("%Y")
      TimeVF = ValidityDateF.strftime("%H")

      title_plot1 = f'ERA5-ecPoint {acc}-hourly tp (mm/{acc} h), Proability of exceeding {rp}-year return period'
      title_plot2 = f'VT: {DayVS} {MonthVS} {yearVS} {TimeVS} UTC - {DayVF} {MonthVF} {yearVF} {TimeVF} UTC'         

      # Plotting the probability
      coastlines = mv.mcoast(
            map_coastline_colour = "charcoal",
            map_coastline_thickness = 2,
            map_coastline_resolution = "full",
            map_coastline_sea_shade = "on",
            map_coastline_sea_shade_colour = "rgb(0.665,0.9193,0.9108)",
            map_boundaries = "on",
            map_boundaries_colour = "charcoal",
            map_boundaries_thickness = 4,
            map_grid_latitude_increment = 10,
            map_grid_longitude_increment = 20,
            map_label_right = "off",
            map_label_top = "off",
            map_label_colour = "charcoal",
            map_grid_thickness = 1,
            map_grid_colour = "charcoal",
            map_label_height = 0.7
            )

      geo_view = mv.geoview(
            map_projection = "epsg:3857",
            map_area_definition = "corners",
            area = mask_domain,
            coastlines = coastlines
            )

      contouring = mv.mcont(
            legend = "on",
            contour = "off",
            contour_level_selection_type = "level_list",
            contour_level_list = [0,0.5,1.5,2.5,3.5,4.5,6.5,8.5,10.5,13.5,16.5,20.5,25.5,30.5,35.5,40.5,50.5,60.5,70.5,80.5,90.5,100],
            contour_label = "off",
            contour_shade = "on",
            contour_shade_colour_method = "list",
            contour_shade_method = "area_fill",
            contour_shade_colour_list = ["white","RGB(0.61,0.91,0.95)","RGB(0.091,0.89,0.99)","RGB(0.015,0.7,0.81)","RGB(0.031,0.55,0.62)","RGB(0.025,0.66,0.24)","RGB(0.015,0.81,0.28)","RGB(0.13,0.99,0.42)","RGB(0.8,0.99,0.13)","RGB(0.65,0.83,0.013)","RGB(0.51,0.64,0.026)","RGB(0.78,0.35,0.017)","RGB(0.92,0.4,0.0073)","RGB(0.99,0.5,0.17)","RGB(0.97,0.65,0.41)","RGB(0.96,0.47,0.54)","RGB(0.98,0.0038,0.1)","RGB(0.88,0.45,0.96)","RGB(0.87,0.26,0.98)","RGB(0.7,0.016,0.79)","RGB(0.52,0.032,0.59)"]
            )

      legend = mv.mlegend(
            legend_text_colour = "charcoal",
            legend_text_font_size = 0.5,
            )

      title = mv.mtext(
            text_line_count = 3,
            text_line_1 = title_plot1,
            text_line_2 = title_plot2,
            text_line_3 = " ",
            text_colour = "charcoal",
            text_font_size = 0.75
            )

      # Saving the maps
      dir_out_temp = git_repo + "/" + dir_out + "/" + base_date.strftime("%Y%m%d") + "_" + base_date.strftime("%H") + "_" + f"{step_f:03d}"
      if not os.path.exists(dir_out_temp):
            os.makedirs(dir_out_temp)
      file_out = dir_out_temp + "/prob_exceed_" + str(rp) + "_rp"  + ".grib"
      png = mv.png_output(output_width = 5000, output_name = file_out)
      mv.setoutput(png)
      mv.plot(geo_view, prob_bitmap, contouring, legend, title)