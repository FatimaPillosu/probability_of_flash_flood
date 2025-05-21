import os
from datetime import datetime, timedelta
import metview as mv

###########################################################################################
# CODE DESCRIPTION
# 11_map_rel_freq_point_ff_per_grid.py creates a map plot of the relative 
# frequency of point accumulated flash flood reports in each grid-box of the domain.

# Usage: python3 11_map_rel_freq_point_ff_per_grid.py

# Runtime: ~ 3 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year_s (integer): start year to consider.
# year_f (integer): final year to consider.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the accumulated point flash flood reports per grid-box.
# dir_out (string): relative path of the directory containing the map plot of the relative frequency.

###########################################################################################
# INPUT PARAMETERS
year_s = 2001
year_f= 2024
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/03_grid_acc_reports_ff"
dir_out = "data/plot/11_map_rel_freq_point_ff_per_grid"
###########################################################################################


# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)
mask = mv.bitmap(mask,0) # bitmap the values outside the domain

# Defining the accumulation periods to consider
the_date_start_s = datetime(year_s,1,1)
the_date_start_f = datetime(year_f,12,31)

# Defining the total number of days in the considered period
num_acc_periods = ((the_date_start_f - the_date_start_s).days + 1)

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

# Bitmapping all those grid-boxes that have never seen any flash flood reports
abs_freq_grid_ff = mv.bitmap(abs_freq_grid_ff, 0)

# Computing the relative frequency
rel_freq_grid_ff = abs_freq_grid_ff / num_acc_periods * 100

# Creating the regions
rel_freq_grid_ff_nw = mv.bitmap(mv.mask(rel_freq_grid_ff, [50,-140,38,-100]) * rel_freq_grid_ff,0)
rel_freq_grid_ff_sw = mv.bitmap(mv.mask(rel_freq_grid_ff, [38,-140,20,-100]) * rel_freq_grid_ff,0)
rel_freq_grid_ff_ne = mv.bitmap(mv.mask(rel_freq_grid_ff, [50,-100,38,-60]) * rel_freq_grid_ff,0)
rel_freq_grid_ff_se = mv.bitmap(mv.mask(rel_freq_grid_ff, [38,-100,20,-60]) * rel_freq_grid_ff,0)

# Ploting the map with the flood reports density
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

contouring_nw = mv.mcont(
    legend = "on", 
    contour = "off",
    contour_level_selection_type = "level_list",
    contour_level_list = [0, 0.1, 1, 10],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_technique = "grid_shading",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = [
        "rgb(0.9504,0.8372,0.6417)",
        "rgb(0.9333,0.6392,0.1255)",
        "rgb(0.3451,0.2387,0.0549)",
        ]
    )

contouring_sw = mv.mcont(
    legend = "on", 
    contour = "off",
    contour_level_selection_type = "level_list",
    contour_level_list = [0, 0.1, 1, 10],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_technique = "grid_shading",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = [
        "rgb(0.9396,0.947,0.7236)",
        "rgb(0.8304,0.8538,0.3854)",
        "rgb(0.3364,0.3429,0.214)",
        ]
    )

contouring_ne = mv.mcont(
    legend = "on", 
    contour = "off",
    contour_level_selection_type = "level_list",
    contour_level_list = [0, 0.1, 1, 10],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_technique = "grid_shading",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = [
        "rgb(0.7592,0.9427,0.9213)",
        "rgb(0.3512,0.8174,0.763)",
        "rgb(0.05917,0.3487,0.3149)",
        ]
    )

contouring_se = mv.mcont(
    legend = "on", 
    contour = "off",
    contour_level_selection_type = "level_list",
    contour_level_list = [0, 0.1, 1, 10],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_technique = "grid_shading",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = [
        "rgb(0.6962,0.8872,0.9509)",
        "rgb(0.1326,0.7278,0.9262)",
        "rgb(0.04315,0.3372,0.4353)",
        ]
    )

legend = mv.mlegend(
    legend_text_colour = "charcoal",
    legend_text_font_size = 0.3,
    )

title = mv.mtext(
    text_line_count = 2,
    text_line_1 = "Relative frequency [%] of gridded flood reports between " + str(year_s) + " and " + str(year_f),
    text_line_2 = " ",
    text_colour = "charcoal",
    text_font_size = 0.75
    )

# Saving the plot
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
    os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/map_rel_freq_point_ff_per_grid" +  str(year_s) + "_" + str(year_f)
png = mv.png_output(output_width = 5000, output_name = file_out)
mv.setoutput(png)
mv.plot(geo_view, rel_freq_grid_ff_nw, contouring_nw, rel_freq_grid_ff_sw, contouring_sw, rel_freq_grid_ff_ne, contouring_ne, rel_freq_grid_ff_se, contouring_se, legend, title)