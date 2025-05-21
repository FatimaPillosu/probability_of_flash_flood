import os
from datetime import datetime
import metview as mv

#####################################################################################
# CODE DESCRIPTION
# 16_swvl_1m_short_fc.py plots the ERA5's short-range forecasts for soil saturation in the top 1m level 
# at 00 UTC.

# Usage: python3 16_swvl_1m_short_fc.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# the_date (date, in YYYYMMDD format): date to consider.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the percentage of instantaneous soil saturation.
# dir_out (string): relative path of the directory containing the plots of the percentage to soil saturation.

#####################################################################################
# INPUT PARAMETERS
the_date = datetime(2021,9,1)
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/06_swvl_1m_short_fc"
dir_out = "data/plot/16_swvl_1m_short_fc"
#####################################################################################

# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)

# Plotting the water content in the soil (in percentage)
print()
print("Plotting the water content in the soil (in percentage) on: ")
print(" - " + the_date.strftime("%Y-%m-%d") + " at " + the_date.strftime("%H") + " UTC")

# Reading the water content in the soil (in percentage) 
file_in = git_repo + "/" + dir_in + "/" + the_date.strftime("%Y") + "/swvl_1m_perc_" + the_date.strftime("%Y%m%d%H") + ".grib"
swvl_perc = mv.read(file_in)
swvl_perc_mask =  mv.bitmap(mask, 0) * swvl_perc

# Plotting the percentage of soil saturation 
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
      contour_level_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      contour_label = "off",
      contour_shade = "on",
      contour_shade_colour_method = "list",
      contour_shade_method = "area_fill",
      contour_shade_colour_list = ["rgb(1,0.9333,0.8)","rgb(1,0.8667,0.6)","rgb(1,0.8,0.3333)","rgb(1,1,0)","rgb(0.3333,1,0)","rgb(0.2,0.6,0)","rgb(0.2,0.8667,0.7333)","rgb(0.2,0.6667,0.6667)","rgb(0.2,0.4667,0.6667)","rgb(0,0,1)"]
      )

legend = mv.mlegend(
      legend_text_colour = "charcoal",
      legend_text_font_size = 0.5,
      )

title = mv.mtext(
      text_line_count = 3,
      text_line_1 = "Percentage [%] of water content in the top 1m of soil",
      text_line_2 = "at " + the_date.strftime("%Y-%m-%d") + " at " + the_date.strftime("%H") + " UTC",
      text_line_3 = " ",
      text_colour = "charcoal",
      text_font_size = 0.75
      )

# Saving the plot
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
      os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/swvl_1m_" + the_date.strftime("%Y%m%d") + "_" + the_date.strftime("%H")
png = mv.png_output(output_width = 5000, output_name = file_out)
mv.setoutput(png)
mv.plot(geo_view, swvl_perc_mask, contouring, legend, title)