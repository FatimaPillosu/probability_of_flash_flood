import os
import metview as mv
import numpy as np

###########################################################################################
# CODE DESCRIPTION
# 04_plot_slor_era5.py plots the slope of the sub-gridscale orography from ERA5 (at 31 km resolution).
# Runtime: negligible.

# INPUT PARAMETERS DESCRIPTION
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# file_in (string): relative path of the file containing the slope of the sub-grid scale orography.
# dir_out (string): relative path of the directory containing the plot for the slope of the sub-grid scale orography.

# INPUT PARAMETERS
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_era5/mask.grib"
file_in = "data/raw/reanalysis/era5/orography/slor.grib"
dir_out = "data/plot/04_slor_era5"
###########################################################################################


# Retrieving the slope of sub-gridscale orography
print("Reading the slope of the sub-gridscale orography...")
slor = mv.read(git_repo + "/" + file_in) * 1

# Reading the USA mask
print("Reading the USA domain ...")
mask = mv.read(git_repo + "/" + file_in_mask)
mask = mv.bitmap(mask,0) # bitmap the values outside the domain

# Selecting the slope of the sub-gridscale orography values within the considered domain
print("Selecting the slope of the sub-gridscale orography values within the considered domain...")
slor_mask = (mask == 1) * slor

# Plotting the slope of the sub-gridscale orography values within the considered domain
print("Plotting the slope of the sub-gridscale orography values within the considered domain...")

coastlines = mv.mcoast(
    map_coastline_colour = "charcoal",
    map_coastline_thickness = 2,
    map_coastline_resolution = "full",
    map_coastline_sea_shade = "on",
    map_coastline_sea_shade_colour = "rgb(0.665,0.9193,0.9108)",
    map_boundaries = "on",
    map_boundaries_colour = "charcoal",
    map_boundaries_thickness = 2,
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
    contour_level_list = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_method = "area_fill",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = [
        "rgb(0.06288,0.1646,0.129)", # 0 - 0.0025
        "rgb(0.05296,0.2764,0.1982)", # 0.0025 - 0.005
        "rgb(0.04558,0.4015,0.2769)", # 0.005 - 0.01
        "rgb(0.03592,0.5366,0.3614)", # 0.01 - 0.02
        "rgb(0.02405,0.6975,0.4618)", #  0.02 - 0.03
        "rgb(0.01006,0.8448,0.5527)", #  0.03 - 0.04
        "rgb(0.4634,0.9641,0.7888)", #  0.04 - 0.05

        "rgb(0.9361,0.9429,0.8061)", #  0.05 - 0.06
        "rgb(0.9393,0.9583,0.579)", #  0.06 - 0.07
        "rgb(0.9843,0.9203,0.2157)" #  0.07 - 0.08
        "rgb(0.9843,0.805,0.2157)", #  0.08 - 0.09
        "rgb(0.9843,0.7153,0.2157)",  #  0.09 - 0.1

        "rgb(0.8371,0.5476,0.009964)", #  0.1 - 0.11
        "rgb(0.6129,0.409,0.03026)",  #  0.11 - 0.12
        "rgb(0.3091,0.219,0.05164)" #  0.12 - 0.13
        ])

legend = mv.mlegend(
    legend_text_colour = "charcoal",
    legend_text_font_size = 0.5,
    )

title = mv.mtext(
    text_line_count = 2,
    text_line_1 = "Slope of the sub-gridscale orography [-]",
    text_line_2 = " ",
    text_colour = "charcoal",
    text_font_size = 0.75
    )

# Saving the plot
print("Saving the map plot ...")
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
    os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/slor" 
png = mv.png_output(output_width = 5000, output_name = file_out)
mv.setoutput(png)
mv.plot(geo_view, slor_mask, contouring, legend, title)