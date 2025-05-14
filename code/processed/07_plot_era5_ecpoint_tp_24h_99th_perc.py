import os
from datetime import datetime, timedelta
import metview as mv

#############################################################################
# CODE DESCRIPTION
# 07_plot_era5_ecpoint_tp_24h_99th_perc.py plots the 99th percentile of 24-hourly rainfall 
# from ERA5-ecPoint.
# Runtime: the code takes up to 90 minutes per year.

# INPUT PARAMETERS DESCRIPTION
# the_date (date, in YYYYMMDD format): date to consider.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the ERA5-ecPoint's 24-hourly rainfall.
# dir_out (string): relative path of the directory containing the 99th percentile's plots.

# INPUT PARAMETERS
the_date = datetime(2021,9,1)
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_era5/mask.grib"
dir_in = "data/raw/reanalysis/era5_ecpoint/tp_24h"
dir_out = "data/plot/07_era5_ecpoint_tp_24h_99th_perc"
#############################################################################


# Defining the forecasts to plot
acc = 24 # accumulation period considered
perc = 99 # percentile considered

# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)
mask_vals = mv.values(mask)
mask_bitmap = mv.bitmap(mask, 0)

# Plotting the 99th percentile for ERA5_ecPoint's 24-hourly rainfall
print()
print("Plotting the 99th percentile for ERA5_ecPoint's 24-hourly rainfall ending on:")

the_date_time_final = the_date + timedelta(hours=24)
print(the_date_time_final.strftime("%Y-%m-%d") + " at " + the_date_time_final.strftime("%H") + " UTC")

tp = mv.read(git_repo + "/" + dir_in + "/" + the_date.strftime("%Y%m") + "/Pt_BC_PERC_" + the_date.strftime("%Y%m%d") + "_024.grib2")
tp_perc = tp[perc-1]
tp_perc_bitmap = mv.bitmap(tp_perc, mask_bitmap)

# Defining the forecasts' valid times
ValidityDateS = the_date_time_final - timedelta(hours=acc)
DayVS = ValidityDateS.strftime("%d")
MonthVS = ValidityDateS.strftime("%B")
yearVS = ValidityDateS.strftime("%Y")
TimeVS = ValidityDateS.strftime("%H")
ValidityDateF = the_date_time_final
DayVF = ValidityDateF.strftime("%d")
MonthVF = ValidityDateF.strftime("%B")
yearVF = ValidityDateF.strftime("%Y")
TimeVF = ValidityDateF.strftime("%H")
title_plot1 = "ERA5-ecPoint " + str(acc) + "-hourly tp (mm/" + str(acc) + "h), " + str(perc) + "th percentile"
title_plot2 = "VT: " + DayVS + " " + MonthVS + " " + yearVS + " " + TimeVS + " UTC - " + DayVF + " " + MonthVF + " " + yearVF + " " + TimeVF  + " UTC"          

# Plot the rainfall climatology
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
      contour_level_list = [0,0.5,2,5,10,20,30,40,50,60,80,100,125,150,200,300,500,5000],
      contour_label = "off",
      contour_shade = "on",
      contour_shade_colour_method = "list",
      contour_shade_method = "area_fill",
      contour_shade_colour_list = ["white","RGB(0.75,0.95,0.93)","RGB(0.45,0.93,0.78)","RGB(0.07,0.85,0.61)","RGB(0.53,0.8,0.13)","RGB(0.6,0.91,0.057)","RGB(0.9,1,0.4)","RGB(0.89,0.89,0.066)","RGB(1,0.73,0.0039)","RGB(1,0.49,0.0039)","red","RGB(0.85,0.0039,1)","RGB(0.63,0.0073,0.92)","RGB(0.37,0.29,0.91)","RGB(0.04,0.04,0.84)","RGB(0.042,0.042,0.43)","RGB(0.45,0.45,0.45)"]
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
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
      os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/tp" + str(acc) + "h_" + str(perc) + "th_" + the_date_time_final.strftime("%Y%m%d") + "_" + the_date_time_final.strftime("%H")
png = mv.png_output(output_width = 5000, output_name = file_out)
mv.setoutput(png)
mv.plot(geo_view, tp_perc_bitmap, contouring, legend, title)